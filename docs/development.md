**English** &nbsp;&nbsp; [简体中文](development.zh-CN.md)

# mJev Developer Guide

[Back to README](../README.md) &nbsp;&nbsp; [Contributing](../CONTRIBUTING.md)

## Entry points and responsibilities

| Behavior to change | Implementation |
| --- | --- |
| HF CLI, input and output | [`demo_hf.py`](../demo_hf.py) |
| Official template, candidates and label validation | [`prompt.py`](../mjev/prompt.py) |
| HF forward, media processing and cache interface | [`hf.py`](../mjev/hf.py) |
| HF question batches and cache branches | [`hf_batch.py`](../mjev/hf_batch.py) |
| vLLM requests and scoring | [`engine.py`](../mjev/engine.py) |
| vLLM attention and cache hooks | [`patch.py`](../mjev/patch.py) |
| Decisions and tie handling | [`decision.py`](../mjev/decision.py) |
| Document adapter, paired evaluation and RLCD | [`docjev/`](../docjev/) &nbsp;&nbsp; [document guide](docjev/README.md) |

Use **mJev** in documentation, `mjev` for Python imports and `MJEV_` for project environment variables.

## Local development

Run from the repository root. See the [HF guide](hf.md) for model inference environment requirements.

```bash
python -m pip install -e '.[docjev,test]'
bash scripts/test.sh -q
```

HF tests use tiny random-weight models to check mechanisms; they do not measure the official 30B model's accuracy. See the [vLLM guide](vllm.md) for CPU and GPU checks in the reference environment. Confirm GPU availability before running GPU checks.

## Implementation constraints

Preserve official weight loading, processors, chat templates and multimodal positional encoding. Do not apply image-only position logic to audio or video. Validate that candidate labels are single tokens at the actual template boundary, and normalize probabilities only over the supplied candidates.

When changing caching, check the full prefix, media parameters, MRoPE, mask compatibility and branch independence. HF currently copies branch KV; vLLM has a different scheduling and cache mechanism. Validation of one backend does not establish correctness of the other.

## Validation and reporting

Choose mechanism checks appropriate to the change, then determine whether a full-weight regression is needed. For mask, cache or numerics changes, compare serial, batch, cached and reordered execution under the same configuration. Preserve raw logits, probability differences, decision changes and timing boundaries. Matching final answers alone does not establish numerical consistency.

See [stability records](stability.md), [paired evaluation](https://github.com/SoMarkAI/mJev/blob/9ff8af6e91c46c04b2984b1a05f637ecb599ed7a/docs/paired_backends.md) and [release validation](https://github.com/SoMarkAI/mJev/blob/9ff8af6e91c46c04b2984b1a05f637ecb599ed7a/docs/validation.md) for evidence and limitations. New results should identify the date, version, sample scope, numerical precision and failures.

## Distinguish two kinds of isolation

- **Independent question branches:** different questions about the same media use separate requests or batch rows. They can reuse the shared prefix but do not read other question branches.
- **Candidate isolation within a question:** when `isolated` is explicitly selected, each candidate reads the shared context, question and its own preceding tokens. The final answer position reads all candidates.

These mechanisms are distinct. The main HF and vLLM APIs and CLIs default to `causal`; candidate isolation requires explicit selection. Candidate isolation provides a controllable attention structure for comparing candidate interactions and exploring future targeted training. Its effect on accuracy or scalability still requires training and ablation evidence.

Structural input validation lives in `mjev/inputs.py` and is shared by preflight, CLI and scoring interfaces. Grouped evaluation saves each completed group in `groups/`, updating `progress.json`, predictions and raw outputs as it proceeds. Later failures preserve completed groups; automatic resume is not currently supported.


<a id="tests"></a>


Commands outside Docker assume an activated Python 3.11+ virtual environment: create it with `python3 -m venv .venv`, then run `source .venv/bin/activate`. Use `python` for installation and execution in that environment; Docker commands use `python3`. Shell scripts also accept `PYTHON=/path/to/venv/bin/python`. Historical execution records retain their original commands.

## Test commands

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

Inspect collection without running tests with `bash scripts/test.sh --collect-only -q`. `--suite` filters before imports, so a base run does not require optional HF/runtime/vLLM modules. Do not use unittest discovery as the repository test gate: it misses function tests and nested suites. Actual results and unavailable checks are in [the validation index](development.md#validation-evidence).

### Document adapter checks

`tests/docjev/` is part of the base suite. It covers document input/reference separation, grouped bilingual splitting, candidate order remapping, RLCD rewards, collective padding and checkpoint/resume metadata. `python -m docjev.compare --help` exposes the paired checkpoint runner; a real pretrained model comparison is a separate GPU command in [the document evaluation guide](docjev/evaluation.md).

<a id="validation-evidence"></a>

## Retained evidence

CPU mechanism checks do not establish pretrained accuracy. Raw historical reports remain in [validation_evidence/](validation_evidence/); they retain their original models and settings, not the current release defaults. Current document scores are in [results](docjev/results.md); cache measurements are in [cache scaling](cache_scaling.md).
