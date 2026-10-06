**English** · [简体中文](development.zh-CN.md)

# mJev Developer Guide

[Back to README](../README.md) · [Contributing](../CONTRIBUTING.md)

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
| Document adapter, paired evaluation and RLCD | [`docjev/`](../docjev/) · [document guide](docjev/README.md) |

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

See [stability records](stability.md), [paired evaluation](paired_backends.md) and [release validation](validation.md) for evidence and limitations. New results should identify the date, version, sample scope, numerical precision and failures.

## Reproduction and current validation

- [Public mini evaluation: prepare → infer → report](reproduce.md)
- [Current validation and backend boundaries](validation_current.md)
- [pytest suites and test entry points](testing.md)
- [Isolated installation checks and limitations](installation_validation.md)

## Distinguish two kinds of isolation

- **Independent question branches:** different questions about the same media use separate requests or batch rows. They can reuse the shared prefix but do not read other question branches.
- **Candidate isolation within a question:** when `isolated` is explicitly selected, each candidate reads the shared context, question and its own preceding tokens. The final answer position reads all candidates.

These mechanisms are distinct. The main HF and vLLM APIs and CLIs default to `causal`; candidate isolation requires explicit selection. Candidate isolation provides a controllable attention structure for comparing candidate interactions and exploring future targeted training. Its effect on accuracy or scalability still requires training and ablation evidence.

Structural input validation lives in `mjev/inputs.py` and is shared by preflight, CLI and scoring interfaces. Grouped evaluation saves each completed group in `groups/`, updating `progress.json`, predictions and raw outputs as it proceeds. Later failures preserve completed groups; automatic resume is not currently supported.
