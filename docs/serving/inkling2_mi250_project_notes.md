# Why this Inkling 2 Small fork targets MI250

These notes summarise the project conversation on 22 November 2026. For commands, checkpoint revision and runtime settings, see the [MI250 serving guide](inkling2_mi250.md).

## What were we trying to run?

The original request was to run Inkling 2 Small on as many AMD Instinct MI250s as necessary, check which existing projects had already solved parts of the problem, and prefer vLLM. The follow-up prioritised serving many simultaneous users and asked for an actual fork. Ten concurrent clients became the first load-test target. Acceptable latency and the final replica count are still open.

## Why start from the MI355X work?

The author of [vLLM PR 67685](https://github.com/vllm-project/vllm/pull/67685) reported serving the all-MXFP8 model with continuous batching on four MI355X GPUs. That work supplied the model integration, attention, cache handling and distributed serving code. Its expert kernels target CDNA4. The MI250 uses CDNA2, and the selected [official checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Small) contains both NVFP4 and MXFP8 expert weights. Adding cards provides memory but leaves both compatibility problems to solve.

The fork starts at PR revision `a6aa3ad9e8b1a7f2c5800b41156f621a2390c62a`. The [GitHub implementation commit](https://github.com/n1024829/vllm-inkling2-mi250/commit/8ca652d60e84611521a0ab1ad94aa526448702b6) has the same Git tree as commit `f08ccaa018fe2ecf060ecc34bef78fa7bbf0eb1a` in the original download. It adds a gfx90a path that keeps expert weights packed, decodes tiles for 16-bit matrix operations, and groups routed tokens from multiple requests by local expert.

## How should the cards be shared?

The first configuration gives one vLLM replica four physical MI250s. ROCm exposes eight 64 GiB GPU dies on this node. Attention and expert weights are distributed across the eight dies, and vLLM continuously batches incoming requests. The starting profile admits 16 active sequences with a 4,096-token context. Extra four-card nodes can host independent replicas behind a shared endpoint. The choice between one replica and several depends on observed queueing and latency.

## What evidence did we have?

| Check | Result at the end of the conversation |
| --- | --- |
| Compile for gfx90a | Eighteen kernel configurations cross-compiled offline, including expert projections and selected attention and router configurations. |
| Check real checkpoint data | All 384 sampled outputs from expert zero in layers 39 and 40 met the reference tolerance. These layers straddle the NVFP4/MXFP8 transition. Largest absolute difference was 0.015625. The serving guide records the interpreter adjustment used for BF16. |
| Load and generate on four MI250s | Pending. The development environment had no MI250 device. Full-model loading, inter-die communication and generation quality require a hardware run. |
| Serve ten users | Pending. The repository includes a sweep at 1, 4, 10, 16 and 32 concurrent clients. The 16-sequence setting is a scheduler limit. Sustainable capacity is unmeasured. |

## What would establish readiness?

Start with one four-card replica. Run the packed-projection checks on hardware, then compare batched and isolated generation across the 1,024-token sliding window. Compare outputs against a reference serving the same mixed checkpoint. Once those checks pass, run the concurrency sweep and record failed requests, aggregate output tokens per second, and p95 time to first token and time per output token at ten clients. Use those measurements and representative prompts to size the deployment.
