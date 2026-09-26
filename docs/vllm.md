**English** · [简体中文](vllm.zh-CN.md)

# vLLM Deployment and Validation

[Back to mJev](../README.md)

## vLLM requirements and installation

The supported reference environment is Linux x86-64, NVIDIA CUDA, **4 × 24 GB NVIDIA GPUs**, BF16 tensor parallelism 4. This paragraph describes the Omni reference; Qwen3-VL-4B defaults to TP=1 and supports an explicit `--tensor-parallel-size` override. See [model selection](models.md). Other hardware configurations have not been validated. The model is approximately 70.5 GB; allow additional disk space for the Docker image and download cache.

Requires Docker with NVIDIA Container Toolkit. The image pins vLLM **0.25.1**, PyTorch **2.11.0+cu130** and Transformers **5.13.1**. The hooks intentionally reject other vLLM versions. An ordinary stock vLLM server does not implement this custom attention mask.

From the repository root:

```bash
# Install the download client in your own Python environment.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
export MODEL_DIR="$HOME/models/Qwen3-Omni-30B-A3B-Instruct"
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir "$MODEL_DIR"
docker build -t mjev:0.1.0 .
mkdir -p outputs
```

The Dockerfile pins the upstream container digest and installs this package with `--no-deps`; it uses the dependencies already in that image. For development inside the same reference environment, run `python3 -m pip install --no-deps --no-build-isolation .` and use the commands below with `python3 demo.py` directly. Run from the checkout: the source `sitecustomize.py` is required for worker bootstrap. Installing the package alone outside this checkout is not a supported launch method.

## Run the vLLM reference

First validate inputs, the official processor/template and label tokens without loading model weights into GPUs:

```bash
docker run --rm \
  -v "$MODEL_DIR:/model:ro" -v "$PWD/outputs:/app/outputs" \
  mjev:0.1.0 --model /model --input examples/multiple.json --check-only
```

This is a preflight check, **not inference**. For actual scoring, reserve four available GPUs:

```bash
docker run --rm --gpus all --ipc=host -e VLLM_BATCH_INVARIANT=1 \
  -v "$MODEL_DIR:/model:ro" -v "$PWD/outputs:/app/outputs" \
  mjev:0.1.0 --model /model --input examples/single.json \
  --mode isolated --output outputs/isolated.json
```

The recommended launch enables stable numerical kernels with `VLLM_BATCH_INVARIANT=1`; keep this setting for batch/cache comparisons. See [numerical profiles](stability.md). Use `--mode causal` for the ordinary causal comparison. Use `--input examples/multiple.json` for concurrent questions on the same image. Common prefix blocks can be reused across requests; concurrent cold requests are not guaranteed to hit each other's cache. Candidate-dependent cache blocks include mask metadata to prevent reuse across incompatible masks. Output reports `num_cached_tokens`.

For your own input, mount its directory read-only at `/inputs` and pass `--input /inputs/task.json`. Image paths resolve relative to the input JSON file.

```json
{
  "image": "rectangle.png",
  "context": "Inspect the supplied picture.",
  "question": "What color is the rectangle?",
  "candidates": ["Red", "Blue", "Green"]
}
```

Multi-question input replaces `question` and `candidates` with a `questions` array containing those two fields per question; image and context remain shared. See `examples/multiple.json`.

Output is a JSON array, one object per question. Each object contains `question`, `mode`, `candidates`, `decision`, `probability_sum`, `num_cached_tokens`, `prompt_tokens` and diagnostic spans. The following is a **schema illustration with invented numbers, not a model result**:

```json
{
  "candidates": [
    {"label": "A", "text": "Red", "token_id": 32, "raw_logit": 2.0, "probability": 0.7310586},
    {"label": "B", "text": "Blue", "token_id": 33, "raw_logit": 1.0, "probability": 0.2689414}
  ],
  "decision": {
    "label": "A", "text": "Red", "token_id": 32,
    "raw_logit": 2.0, "probability": 0.7310586,
    "index": 0, "tied_labels": ["A"], "tie_policy": "first_in_input_order"
  },
  "probability_sum": 1.0
}
```

Candidate count may vary from 2 to 128, subject to exact single-token label validation and the context limit. Counts up to 30 were exercised with real inference and up to 52 with tokenizer checks; 128 is an input bound, not a promise that every label passes. Duplicate/empty choices, embedded chat control tokens and labels spanning multiple tokens are rejected.

## Tests and validation status

CPU mask/decision tests in the built image:

```bash
docker run --rm --entrypoint bash mjev:0.1.0 scripts/test.sh
```

Real-model tests (four free GPUs; model initialization can take much longer than an individual request):

```bash
docker run --rm --gpus all --ipc=host --entrypoint bash \
  -e MJEV_MODEL=/model \
  -v "$MODEL_DIR:/model:ro" -v "$PWD/outputs:/app/outputs" \
  mjev:0.1.0 scripts/test_gpu.sh
```

The GPU suite checks mask isolation with hidden-state interventions, a causal positive control, comparison to stock Triton attention, candidate permutations, concurrent questions, probability sums and mask-aware prefix-cache reuse. It writes `outputs/e2e/runs/<run>/_SUCCESS` only after every assertion passes. Reordering choices need not preserve predictions: original position encoding and label effects remain.

The pre-release prototype passed this GPU suite on the reference environment. Release packaging is checked separately; see [validation](validation.md) for exact scope. We do not claim a new GPU run where only CPU/preflight checks were performed.

## Scope and limitations

- Thinker only; Talker is not instantiated. HF and the pooling backend support image, audio, video and video with audio.
- Official multimodal processor, chat template, vision encoder, position encoding, scheduler and paged KV cache are retained.
- The custom attention path gathers cached K/V into dense SDPA tensors. This is a correctness prototype, not an optimized attention kernel or production serving stack.
- Eager execution, no chunked prefill, no asynchronous scheduling, unquantized KV, maximum sequence length 4096, at most 8 scheduled sequences, image maximum 262144 pixels. These settings are in `mjev/engine.py`.
- Small text in document images may be lost at this image resolution. No training code or weights are provided. The optional HTTP runtime is a separate experimental implementation.
