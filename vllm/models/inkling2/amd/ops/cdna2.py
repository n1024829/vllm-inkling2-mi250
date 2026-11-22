# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed Inkling experts for CDNA2. The MFMA operands are 16-bit floats."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _e4m3(bits):
    mag = bits.to(tl.int32) & 127
    e = mag >> 3
    m = mag & 7
    power = ((e + 120) << 23).to(tl.float32, bitcast=True)
    value = tl.where(e == 0, m * (1.0 / 512), power * (1 + m * 0.125))
    value = tl.where(mag == 127, float("nan"), value)
    return tl.where((bits & 128) != 0, -value, value)


@triton.jit
def _round_e4m3(x):
    # RNE on the E4M3FN grid, including its subnormal interval.
    mag = tl.minimum(tl.abs(x), 448.0)
    exponent = (mag.to(tl.int32, bitcast=True) >> 23) & 255
    step = (tl.maximum(118, exponent - 3) << 23).to(tl.float32, bitcast=True)
    scaled = mag / step
    lo = tl.floor(scaled)
    frac = scaled - lo
    up = (frac > 0.5) | ((frac == 0.5) & ((lo.to(tl.int32) & 1) != 0))
    rounded = (lo + up.to(tl.float32)) * step
    return tl.where(x < 0, -rounded, rounded)


@triton.jit
def _round_e2m1(x):
    mag = tl.abs(x)
    q = tl.where(mag > 0.25, 0.5, 0.0)
    q = tl.where(mag >= 0.75, 1.0, q)
    q = tl.where(mag > 1.25, 1.5, q)
    q = tl.where(mag >= 1.75, 2.0, q)
    q = tl.where(mag > 2.5, 3.0, q)
    q = tl.where(mag >= 3.5, 4.0, q)
    q = tl.where(mag > 5.0, 6.0, q)
    return tl.where(x < 0, -q, q)


@triton.jit
def _e2m1(bits):
    mag = bits & 7
    value = tl.where(mag < 4, mag * 0.5, tl.where(mag < 6, mag - 2, 2 * mag - 8))
    return tl.where((bits & 8) != 0, -value, value)


@triton.jit
def _quantize_rows(X, Out, RowMax, K: tl.constexpr, B: tl.constexpr, NV: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, B)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)
    if NV:
        amax = tl.max(tl.abs(x), 0)
        tl.store(RowMax + row, amax)
        enc = tl.minimum(tl.div_rn(1536.0, amax), 3.4028234663852886e38)
        dec = tl.div_rn(1.0, enc)
        outer = amax * (1.0 / 1536.0)
        groups = tl.reshape(x, (B // 16, 16))
        peak = tl.max(tl.abs(groups), 1)
        s6 = _round_e4m3(peak * (enc * (1.0 / 6.0)))
        s4 = _round_e4m3(peak * (enc * 0.25))
        inv6 = tl.minimum(tl.div_rn(1.0, s6 * dec), 3.4028234663852886e38)
        inv4 = tl.minimum(tl.div_rn(1.0, s4 * dec), 3.4028234663852886e38)
        r6 = _round_e2m1(groups * inv6[:, None]) * s6[:, None]
        r4 = _round_e2m1(groups * inv4[:, None]) * s4[:, None]
        err6 = tl.abs(r6 * outer - groups)
        err4 = tl.abs(r4 * outer - groups)
        # Same adjacent-pair MAE reduction and tie rule as the checkpoint.
        for width in tl.static_range(3, -1, -1):
            a, b = tl.split(tl.reshape(err6, (B // 16, 1 << width, 2)))
            err6 = a + b
            a, b = tl.split(tl.reshape(err4, (B // 16, 1 << width, 2)))
            err4 = a + b
        result = tl.where(tl.reshape(err4 < err6, (B // 16, 1)), r4, r6)
    else:
        groups = tl.reshape(x, (B // 32, 32))
        peak = tl.max(tl.abs(groups), 1)
        raw = peak * (1.0 / 448.0)
        e = (raw.to(tl.int32, bitcast=True) + 0x007FFFFF) >> 23
        e = tl.where(raw <= tl.full((), 2.0**-127, tl.float32), 0, tl.minimum(e, 254))
        scale = tl.maximum(
            (e << 23).to(tl.float32, bitcast=True), tl.full((), 2.0**-127, tl.float32)
        )
        inv = tl.where(e == 0, tl.full((), 2.0**127, tl.float32), tl.div_rn(1.0, scale))
        result = _round_e4m3(groups * inv[:, None]) * scale[:, None]
    tl.store(Out + row * K + cols, tl.reshape(result, (B,)), cols < K)


@triton.jit
def _grouped_mm(
    A,
    W,
    S,
    RowMax,
    WeightMax,
    Den,
    Sorted,
    Experts,
    Padded,
    Out,
    R,
    K: tl.constexpr,
    N: tl.constexpr,
    REPEAT: tl.constexpr,
    NV: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    block = tl.program_id(0)
    if block * BM >= tl.load(Padded):
        return
    e = tl.load(Experts + block).to(tl.int64)
    if e < 0:
        return
    r = tl.load(Sorted + block * BM + tl.arange(0, BM)).to(tl.int64)
    src = r // REPEAT
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    group: tl.constexpr = 16 if NV else 32
    pack: tl.constexpr = 2 if NV else 1
    for start in range(0, K, BK):
        col = start + k
        a = tl.load(A + src[:, None] * K + col[None, :], r[:, None] < R, other=0)
        wrow = e * N + n[:, None]
        raw = tl.load(W + wrow * (K // pack) + col[None, :] // pack)
        s = tl.load(S + wrow * (K // group) + col[None, :] // group)
        if NV:
            nibble = (raw.to(tl.int32) >> ((col[None, :] & 1) * 4)) & 15
            w = (_e2m1(nibble) * _e4m3(s)).to(tl.float16)
        else:
            exponent = s.to(tl.int32)
            scale = tl.where(
                exponent == 0,
                tl.full((), 2.0**-127, tl.float32),
                (exponent << 23).to(tl.float32, bitcast=True),
            )
            scale = tl.where(exponent == 255, float("nan"), scale)
            w = (_e4m3(raw) * scale).to(tl.bfloat16)
        acc = tl.dot(a, tl.trans(w), acc)
    if NV:
        amax = tl.load(RowMax + src, r < R, other=0)
        outer = (amax * tl.load(WeightMax + e)) * tl.div_rn(1.0, 1536.0 * tl.load(Den))
        acc = acc * outer[:, None]
    tl.store(Out + r[:, None] * N + n[None, :], acc.to(tl.bfloat16), r[:, None] < R)


def project(x, weight, scales, amax, denominator, routing, rows, repeat):
    """Compute both formats in routed-slot order. Remote slots contribute zero."""
    sorted_ids, expert_ids, padded = routing
    m, k = x.shape
    n = weight.shape[1]
    nv = amax is not None
    assert x.is_contiguous() and k % 64 == 0 and n % 32 == 0
    act = torch.empty_like(x, dtype=torch.float16 if nv else torch.bfloat16)
    row_max = x.new_empty(m, dtype=torch.float32)
    _quantize_rows[(m,)](
        x, act, row_max, k, triton.next_power_of_2(k), nv, enable_fp_fusion=False
    )
    out = x.new_zeros((rows, n), dtype=torch.bfloat16)
    # The native aligner allocates for every global expert. Its local mapped
    # tiles are dense at the front, so bound launches using this EP shard.
    active_experts = min(weight.shape[0], rows)
    max_blocks = active_experts + (rows - active_experts) // 16
    _grouped_mm[(min(expert_ids.numel(), max_blocks), n // 32)](
        act,
        weight.view(torch.uint8),
        scales.view(torch.uint8),
        row_max,
        amax if nv else row_max,
        denominator if nv else row_max,
        sorted_ids,
        expert_ids,
        padded,
        out,
        rows,
        k,
        n,
        repeat,
        nv,
        16,
        32,
        64,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out
