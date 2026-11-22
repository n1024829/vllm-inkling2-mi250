# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.inkling2.amd.ops.cdna2 import project
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.is_device_capability(90)),
    reason="CDNA2 expert projection",
)


def round_nv(x):
    # Round each magnitude to its nearest FP4 value; choose even code on ties.
    levels = x.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    distances = (x.abs()[..., None] - levels).abs()
    even = distances[..., ::2].min(-1)
    odd = distances[..., 1::2].min(-1)
    index = torch.where(
        even.values <= odd.values, 2 * even.indices, 2 * odd.indices + 1
    )
    return levels[index] * x.sign()


def activation_reference(x, nv):
    x = x.float()
    if nv:
        amax = x.abs().amax(1)
        blocks = x.unflatten(-1, (-1, 16))
        outer = amax[:, None, None] / 1536
        safe_outer = outer.masked_fill(outer == 0, 1)
        peak = blocks.abs().amax(-1, keepdim=True)
        candidates = []
        for maximum in (6, 4):
            scale = (
                (peak / (safe_outer * maximum))
                .clamp(max=448)
                .to(torch.float8_e4m3fn)
                .float()
            )
            unit = scale * outer
            candidates.append(round_nv(blocks / unit.masked_fill(unit == 0, 1)) * scale)
        six, four = candidates
        better = (four * outer - blocks).abs().sum(-1, keepdim=True) < (
            six * outer - blocks
        ).abs().sum(-1, keepdim=True)
        return torch.where(better, four, six).flatten(1), amax
    blocks = x.unflatten(-1, (-1, 32))
    raw = blocks.abs().amax(-1, keepdim=True) / 448
    scale = raw.clamp_min(2.0**-127).log2().ceil().exp2()
    return ((blocks / scale).to(torch.float8_e4m3fn).float() * scale).flatten(1), None


@pytest.mark.parametrize("nv", [True, False])
@pytest.mark.parametrize("repeat", [1, 2])
def test_packed_projection_with_remote_and_padded_slots(nv, repeat):
    device = "cuda"
    gen = torch.Generator().manual_seed(32)
    r, n, k = 20, 64, 128
    x = torch.randn(r // repeat, k, generator=gen).bfloat16()
    x[1].zero_()
    local_slots = ([0, 4, 6, 9, 12, 15, 17], [1, 2, 7, 10, 18, 19])
    packed_rows = torch.tensor(
        [v for slots in local_slots for v in [*slots, *([r] * (16 - len(slots)))]],
        dtype=torch.int32,
    )
    routing = tuple(
        t.to(device)
        for t in (
            packed_rows,
            torch.tensor([1, 0, -1], dtype=torch.int32),
            torch.tensor([32], dtype=torch.int32),
        )
    )
    if nv:
        weight = torch.randint(256, (2, n, k // 2), dtype=torch.uint8, generator=gen)
        scale = (torch.rand(2, n, k // 16, generator=gen) + 0.2).to(torch.float8_e4m3fn)
        codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
        )
        w = lut[codes] * scale.float().repeat_interleave(16, -1)
        wa, den = torch.tensor([7.0, 11.0]), torch.tensor([16.0])
    else:
        weight = torch.randn(2, n, k, generator=gen).to(torch.float8_e4m3fn)
        scale = torch.randint(
            122, 129, (2, n, k // 32), generator=gen, dtype=torch.uint8
        )
        w = weight.float() * (scale.float() - 127).exp2().repeat_interleave(32, -1)
        wa = den = None
    act, rowmax = activation_reference(x, nv)
    ref = torch.zeros(r, n, dtype=torch.bfloat16)
    for expert, slots in zip((1, 0), local_slots):
        src = torch.tensor(slots) // repeat
        y = act[src].double() @ w[expert].double().T
        if nv:
            y *= (rowmax[src] * wa[expert] / (1536 * den))[:, None]
        ref[slots] = y.bfloat16()
    actual = project(
        x.to(device),
        weight.to(device),
        scale.to(device),
        None if wa is None else wa.to(device),
        None if den is None else den.to(device),
        routing,
        r,
        repeat,
    )
    torch.testing.assert_close(actual.cpu().float(), ref.float(), rtol=0.008, atol=1e-5)


@pytest.mark.distributed(num_gpus=8)
def test_real_checkpoint_mixed_length_batch():
    """Compare batched and isolated next-token probabilities across the local window."""
    import os

    from vllm import LLM, SamplingParams

    path = os.environ.get("INKLING2_CHECKPOINT")
    if not path or torch.accelerator.device_count() != 8:
        pytest.skip("Set INKLING2_CHECKPOINT and expose all eight gfx90a dies")
    engine = LLM(
        model=path,
        tensor_parallel_size=8,
        distributed_executor_backend="mp",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        max_model_len=4096,
        max_num_seqs=16,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0},
    )
    # Every prompt has a distinct suffix to detect cross-request KV/conv state leaks.
    prompts = [
        {"prompt_token_ids": [151 + i, 181 + i] * size + [200 + i]}
        for i, size in enumerate((32, 510, 513, 630))
    ]
    sampling = SamplingParams(temperature=0, max_tokens=1, logprobs=8)
    single = [engine.generate([p], sampling, use_tqdm=False)[0] for p in prompts]
    batch = engine.generate(prompts, sampling, use_tqdm=False)
    for ref, res in zip(single, batch, strict=True):
        ref_probs, res_probs = ref.outputs[0].logprobs, res.outputs[0].logprobs
        assert ref_probs is not None and res_probs is not None
        expected, actual = ref_probs[0], res_probs[0]
        token = ref.outputs[0].token_ids[0]
        assert token in actual
        assert actual[token].logprob == pytest.approx(expected[token].logprob, abs=0.10)
