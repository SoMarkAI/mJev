**English** · [简体中文](testing.zh-CN.md)

Commands outside Docker assume an activated Python 3.11+ virtual environment: create it with `python3 -m venv .venv`, then run `source .venv/bin/activate`. Use `python` for installation and execution in that environment; Docker commands use `python3`. Shell scripts also accept `PYTHON=/path/to/venv/bin/python`. Historical execution records retain their original commands.

# Test entry points

From a fresh Python 3.11+ environment, run `python -m pip install -e '.[docjev,test]'`. This installs pinned torch/Transformers plus pytest, requests and accelerate for tiny CPU checkpoint tests. It is **not** the complete HF media/deployment extra. Run from the checkout:

```bash
bash scripts/test.sh --suite base -q
bash scripts/test.sh --suite hf -q
bash scripts/test.sh --suite runtime -q
bash scripts/test.sh --suite vllm -q
bash scripts/test.sh -q
```

`PYTHON=/path/to/venv/bin/python bash scripts/test.sh ...` selects an interpreter. All entry points use pytest, which discovers both unittest classes and function-style tests recursively. The default `all` includes base, HF, runtime and optional vLLM CPU suites, but never GPU tests. Missing optional vLLM dependencies are reported as **skips**, not successful checks. Direct `python -m pytest tests` uses the same configuration; use the script to explicitly disable runtime hooks.

| Suite | Scope | Exclusions / requirements |
| --- | --- | --- |
| base | Dataset/metric contracts, masks, decisions, report/provenance checks | No real pretrained scoring |
| hf | Tiny random-weight Thinker forwards, cache/batch and checkpoint-loading checks | No official 30B accuracy; accelerate required |
| runtime | Experimental service/Tree-KV pure helpers, mask oracle, projection test | Projection test requires vLLM; not full-model Tree-KV parity |
| vllm | Pooling media placement and prefix-bypass CPU contracts | Requires pinned vLLM environment |
| gpu | Official-model vLLM end-to-end mechanism suite | Explicit opt-in, four free GPUs, local weights |

```bash
MJEV_MODEL=/path/to/official/model bash scripts/test_gpu.sh -q
```

The GPU script invokes pytest's GPU suite, which runs the existing `tests/e2e.py` assertions unchanged. It sets `VLLM_BATCH_INVARIANT=1` unless explicitly overridden. It does not stop services. HF full-weight and concurrent-cache studies remain separate experiment commands in `benchmarks/integrated/`; they are **not** silently included in CPU pytest or claimed by the GPU wrapper. Their local datasets and explicit numerics/revision requirements are documented separately.

Inspect collection without running tests with `bash scripts/test.sh --collect-only -q`. `--suite` filters before imports, so a base run does not require optional HF/runtime/vLLM modules. Do not use unittest discovery as the repository test gate: it misses function tests and nested suites. Actual results and unavailable checks are in [the validation index](validation_current.md).

## Document adapter checks

`tests/docjev/` is part of the base suite. It covers document input/reference separation, grouped bilingual splitting, candidate order remapping, RLCD rewards, collective padding and checkpoint/resume metadata. `python -m docjev.compare --help` exposes the paired checkpoint runner; a real pretrained model comparison is a separate GPU command in [the document evaluation guide](docjev/evaluation.md).
