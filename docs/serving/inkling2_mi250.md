# Inkling 2 Small on MI250

This experimental branch targets the official mixed NVFP4/MXFP8 checkpoint on a single Linux node with four physical MI250 cards. ROCm exposes eight gfx90a dies, each with 64 GiB. vLLM runs one server with TP8 attention and EP8 experts, and continuously batches requests. The default profile admits up to 16 sequences, with 4,096 tokens of context per sequence and 512 scheduled tokens per iteration. These are initial settings for measurement.

## Build and launch

Run the commands in the root of this fork. Use a Linux ROCm host with Docker Buildx, access to the eight devices, and disk space for the build and the roughly 394 GB checkpoint. Keep the Git metadata and the fork release tag when building: the Dockerfile uses git describe to version the wheel. The bake file included in this revision builds its ROCm dependencies. CPU-only build hosts can compile the image. Large builds need substantial disk, RAM and time.

```bash
TAG=inkling2-mi250:dev docker buildx bake -f docker/docker-bake-rocm.hcl \
  --set vllm-openai.args.ARCH=gfx90a vllm-openai

mkdir -p models/Inkling-2-Small
hf download thinkingmachines/Inkling-2-Small \
  --revision 885d78bfca61a7e9a498e33cfa7ad35850533b8c \
  --local-dir models/Inkling-2-Small

recipe="$PWD/examples/online_serving/inkling2_mi250"
docker run --rm --name inkling2-mi250 --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add video \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models/Inkling-2-Small:/model:ro" -e MODEL=/model \
  -v "$recipe:/recipe:ro" --entrypoint bash inkling2-mi250:dev /recipe/serve.sh
```

The launcher checks the visible GPU architecture and die count, then uses RCCL, eager execution, BF16 KV/activations, text input and the mixed checkpoint. Keep the first server on loopback. Route shared clients through the organisation's existing authenticated gateway. Increase context length or batch size only after checking the cache capacity reported at startup. Override settings using Docker environment variables such as `-e MAX_MODEL_LEN=8192` and `-e BATCHED_TOKENS=1024`.

## Validate output and concurrency

From another terminal, send a chat request and inspect its answer before timing synthetic traffic.

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling2-mi250","messages":[{"role":"user","content":"What is the chemical symbol for helium?"}],"temperature":0,"max_tokens":256}'

docker exec -e RESULT_DIR=/tmp/inkling-results inkling2-mi250 bash /recipe/bench.sh
docker cp inkling2-mi250:/tmp/inkling-results ./inkling-results
```

The sweep uses 1, 4, 10, 16 and 32 concurrent clients, random 1,024-token inputs and 128-token outputs. Each JSON file includes aggregate output tokens/s and p50/p95/p99 time to first token, time per output token and end-to-end latency. A sweep with any failed or missing request exits with an error. At 32 clients the default 16-sequence limit deliberately includes queueing. Record the build revision and repeat with representative prompts before choosing capacity. More four-card nodes can each host an independent replica behind the same gateway. Use measured p95 latency at the intended load to choose the number of replicas.

GPU tests in a development checkout with vLLM and its test dependencies installed:

```bash
VLLM_ROCM_USE_AITER=0 .venv/bin/python -m pytest -q tests/models/inkling2/test_cdna2.py -k packed_projection
# Stop the running server to free all eight dies first.
VLLM_ROCM_USE_AITER=0 INKLING2_CHECKPOINT="$PWD/models/Inkling-2-Small" \
  .venv/bin/python -m pytest -q tests/models/inkling2/test_cdna2.py -k real_checkpoint
```

The first test compares both packed formats against independent dequantisation and FP64 reference GEMMs. The second loads the full checkpoint and checks four mixed-length prompts in isolation and in a batch. For model-quality validation, compare log probabilities and a task benchmark against the same pinned mixed checkpoint on a supported reference system.

## Implementation and verification

The parent is [vLLM ROCm PR 67685](https://github.com/vllm-project/vllm/pull/67685) at `a6aa3ad9e8b1a7f2c5800b41156f621a2390c62a`. That branch reports MI355X results using all-MXFP8. This fork reads the official quantisation sidecar to select the expert format at each layer. A grouped Triton GEMM unpacks weight tiles into FP16 for NVFP4 and BF16 for MXFP8, using FP32 accumulation. The NVFP4 path implements four-over-six activation scaling, per-row and per-expert outer scales. Both projections reuse the expert assignment across the current batch. ROCm attention and short-convolution state handling come from the parent.

On a CPU build host the new projections and activation kernels were cross-compiled at the model dimensions for gfx90a, and selected six-head local/global attention kernels were cross-compiled. Offline reference comparisons used synthetic fixtures and the first 32 output rows of expert zero for both projections in real checkpoint layers 39 and 40. Across the four real slices, 384 outputs met the comparison tolerance (relative 0.008, absolute 0.00002); the largest absolute difference was 0.015625. These tests used the Triton interpreter and independent FP64 reference products. The MXFP8 projection comparison required decoding BF16 input words before NumPy matmul because the stock interpreter treats those words as integers. Compiled device code was not executed. Docker image construction, checkpoint loading, RCCL operation, output quality and multi-user performance on MI250 all still require hardware validation. Graph capture, DSpark and vision are outside the initial serving profile.
