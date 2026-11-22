#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
model=${MODEL:-thinkingmachines/Inkling-2-Small}
export VLLM_ROCM_USE_AITER=0
python3 - <<'PY'
import torch
devices = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
if len(devices) != 8 or any(d.gcnArchName.split(":")[0] != "gfx90a" for d in devices):
    raise SystemExit("Expose exactly eight gfx90a GCDs per replica: " + str(devices))
print("Visible gfx90a memory (GiB):", [round(d.total_memory / 2**30, 1) for d in devices])
PY
exec vllm serve "$model" --served-model-name inkling2-mi250 \
  --tensor-parallel-size 8 --distributed-executor-backend mp \
  --dtype bfloat16 --enforce-eager --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":0}' --reasoning-parser inkling \
  --max-model-len "${MAX_MODEL_LEN:-4096}" --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${BATCHED_TOKENS:-512}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --host 0.0.0.0 --port "${PORT:-8000}" "$@"
