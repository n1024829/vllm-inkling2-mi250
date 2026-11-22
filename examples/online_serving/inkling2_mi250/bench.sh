#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
results=${RESULT_DIR:-results/inkling2-mi250-$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$results"
for clients in 1 4 10 16 32; do
  count=$((clients * 8))
  file="c${clients}.json"
  vllm bench serve --base-url "${BASE_URL:-http://127.0.0.1:8000}" \
    --backend openai-chat --endpoint /v1/chat/completions \
    --model inkling2-mi250 --tokenizer "${MODEL:-thinkingmachines/Inkling-2-Small}" \
    --tokenizer-mode inkling --dataset-name random --random-input-len 1024 \
    --random-output-len 128 --ignore-eos --seed 73 --num-warmups 2 \
    --max-concurrency "$clients" --num-prompts "$count" \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir "$results" --result-filename "$file"
  python3 - "$results/$file" "$count" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    result = json.load(f)
if result["completed"] != int(sys.argv[2]) or result["failed"]:
    raise SystemExit(f"Incomplete benchmark: {result['completed']} completed, {result['failed']} failed")
PY
done
