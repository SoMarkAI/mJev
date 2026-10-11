# Development and tests

[简体中文](development.zh-CN.md) · [Project](../README.md) · [Contributing](../CONTRIBUTING.md)

## Code map

| Feature | Location |
|---|---|
| General HF inference | `demo_hf.py`, `mjev/hf.py` |
| Document input, inference and HTTP service | `docjev/io.py`, `docjev/engine.py`, `docjev/service.py` |
| Templates, labels and input validation | `mjev/prompt.py`, `mjev/inputs.py` |
| Batch, prefix cache and numerics | `mjev/hf_batch.py`, `mjev/hf_request_cache.py`, `mjev/hf_numerics.py` |
| vLLM scoring and hooks | `demo.py`, `mjev/engine.py`, `mjev/patch.py` |
| Document evaluation and model comparison | `docjev/evaluate.py`, `docjev/compare.py` |
| General evaluation and cache probe | `mjev/benchmark/`, `benchmarks/integrated/validate_model.py` |

Preserve official weight loading, processors, chat templates and position encoding. Validate labels at the actual template boundary, apply softmax only to the supplied candidates and keep references out of inference inputs. Cache changes require branch independence, mask, position and media checks, comparing raw logits under the same numerical profile.

<a id="tests"></a>

## Tests

Inside a Python 3.11+ virtual environment, from the repository root:

```bash
python -m pip install -e '.[docjev,test]'
bash scripts/test.sh -q
```

Select an interpreter with `PYTHON=/path/to/venv/bin/python bash scripts/test.sh -q`. The default runs all CPU tests; GPU tests are explicit. Select `--suite base`, `--suite hf` or `--suite vllm`; `--collect-only` checks discovery. Missing optional vLLM is reported as a skip.

- base: input and metrics, service scheduling, decisions, document and general evaluation contracts.
- hf: tiny random-weight forwards, batch/cache, numerics and checkpoint loading, not pretrained accuracy.
- vllm: media placement and prefix-bypass contracts with the pinned vLLM installation.
- gpu: official Omni vLLM end-to-end mechanism checks, requiring four free GPUs and complete local weights.

```bash
MJEV_MODEL=/path/to/official/model bash scripts/test_gpu.sh -q
```

[Service tests](../tests/docjev/test_service.py) cover HTTP, queueing and cache scheduling. Real-checkpoint scoring and performance require separate [document](docjev/evaluation.md) or [general evaluation](benchmark.md).

<a id="validation-evidence"></a>

## Evidence

[Document scores](docjev/results.md) and [cache measurements](cache_scaling.md) retain their own sample scopes and settings. Cache raw data and its checker remain in `docs/validation_evidence/`; historical harnesses remain byte-for-byte unchanged. New measurements archive the selected source commit as described in the cache report.

Other historical reports, the experimental HTTP runtime, AV dataset downloaders and RLCD training tools remain in the [fixed pre-cleanup checkout](https://github.com/SoMarkAI/mJev/tree/a313f135041cf893db27a81f40052b9e66b2b5a2). Historical records are not new GPU runs of the current code.
