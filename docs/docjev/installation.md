# Installation

[Project](../../README.md) · [中文首页](../../README.zh-CN.md)

## Supported configuration

| Component | Validated version |
| :--- | :--- |
| Python | 3.12 (package requires 3.11+) |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.13.1 |
| Accelerate | 1.14.0 |
| Safetensors | 0.8.0 |
| Model | Qwen3-VL-4B-Instruct, revision `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96 GB |

The native HF runtime is pinned to Transformers 5.13.1. Other versions are rejected rather than silently running an unvalidated attention/template path. vLLM, FlashAttention builds and FFmpeg are not required for image-only use. The package does not include or download model weights during inference.

Install PyTorch for your hardware before installing the HF extra. The CUDA 13.0 command in the README describes the validated environment; use a driver-compatible PyTorch wheel if your system differs. CUDA availability is a separate check from package import.

```bash
python -m pip install -e '.[docjev]'
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir models/Qwen3-VL-4B-Instruct
```

`hf` is supplied by the Hugging Face Hub dependency. Official model access may require your own Hugging Face authentication. Never put access tokens in input JSON, scripts or commits.

## mjev-doc weights

**mjev-doc** weights will be released on Hugging Face, following the same distribution format as mJev. Model code, documentation and evaluation stay in this repository. The model download link will be added when the checkpoint is published.

To run a checkpoint now, pass its complete local exported directory to `--model`.

Both the base model and exported training checkpoint must contain complete model shards, the safetensors index, configuration, tokenizer, processor and chat template. Pass the directory to `--model`; loaders use `local_files_only=True`. A partial checkpoint will not work.

```bash
python demo_docjev.py --model /path/to/checkpoint-final \
  --input examples/docjev/multiple.json --output outputs/trained-demo.json
```

For a single GPU, `--device-map cuda:0` explicitly places inference there. `CUDA_VISIBLE_DEVICES` controls which devices the process can see. Automatic placement can distribute weights, but multi-device inference performance is not measured by the first release.

## Lightweight tools

For offline probability evaluation, input validation and development without model inference:

```bash
python -m pip install -e '.[test]'
python -m docjev.evaluate --predictions /path/to/predictions.jsonl \
  --output outputs/metrics.json
```

The complete test suite imports the HF and RLCD modules, so install both `hf` and `dev` for all tests. Local CPU kernels are sufficient for unit tests; actual model scoring is GPU-validated separately.

## Troubleshooting

| Symptom | Check |
| :--- | :--- |
| Model/config not found | Download the complete model to the exact local directory. |
| Transformers version error | Use the pinned HF extra in a clean environment. |
| CUDA unavailable | Verify the driver, CUDA-compatible torch wheel and GPU visibility. |
| Out of memory | Use batch size 1, native numerics and no prefix cache; reduce image pixel budget if needed. |
| Input exceeds 4,000 tokens | Reduce image budget or task text explicitly; inputs are never silently truncated. |
| Candidate label not a single token | Reduce candidate count. Alphabetical labels beyond Z are model/template dependent. |
| Cache/batch CLI rejected | Add `--numerics stable`; compare against an uncached stable baseline. |

Changing pixel limits or numerics changes the evaluation protocol. Record it when reporting results.
