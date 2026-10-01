**English** · [简体中文](qwen3_vl_validation.zh-CN.md)

# Qwen3-VL-4B validation — 2026-09-26

Both main backends completed real **Qwen3-VL-4B-Instruct** image/video inference on a single 24 GB NVIDIA GPU each. This is a small integration and consistency validation, not evidence of general benchmark superiority. [Install and reproduce](models.md).

## Environment and provenance

- Official revision: `ebb281ec70b05090aa6165b016eac8ec08e71b17`. Both weight shards independently matched the pinned Hub SHA-256 metadata: [receipts](validation_evidence/qwen3-vl-weight-integrity.json).
- HF: fresh Linux Python 3.11.16 virtualenv, no system-site packages; `.[hf-vl,test]` and the official TorchCodec 0.11.0+cpu wheel. Pip downloads were cached. Neither vLLM nor `qwen_omni_utils` was installed. `pip check`, seeded CUDA computation and 123 CPU tests passed; 3 optional-vLLM modules skipped.
- vLLM: existing pinned 0.25.1 / Transformers 5.13.1 runtime, Python 3.12; 130 CPU tests passed. This was **not** a new vLLM image build. Its decoder version differs from the HF environment; full dependencies are included in each report.
- The documented HF preflight and cached demo were repeated in the newly installed environment; both passed. Installation and model downloading were not repeated.
- Code base: `2e75abf` plus the adapter changes. Reports record actual dirty-tree source hashes and available Git metadata; vLLM's minimal image has no Git executable, so its Git fields are explicitly null. The source hashes remain available. Documentation and the later worker-bootstrap fix must not be retroactively attributed to earlier HF runs.

[Machine-readable validation summary](validation_evidence/qwen3-vl-validation-summary.json).

## What passed

| Check | HF | vLLM |
| --- | --- | --- |
| Full-weight image + video synthetic smoke | 6/6 processed | 6/6 processed |
| Causal and candidate-isolated modes | Passed | Passed |
| Serial / batch / cached-serial / cached-batch | Passed | Passed |
| Cached-token reuse and probability normalization | Passed | Passed |
| Reversed question order | Passed | Passed |
| Reversed candidate mapping | Passed; answer invariance not assumed | Passed; answer invariance not assumed |
| Maximum logit delta in tested execution comparisons | **0.0** | **0.0** |
| NExT-QA original-label video pilot | 18/18 processed | 18/18 processed |

Within-backend cache comparisons use `stable` numerics, full LM-head projection and both attention modes. Each timed configuration has two repetitions after one warmup; every repetition is checked and its comparison is saved, along with the reversed-question comparison. They are small-scope observations, not a guarantee for arbitrary inputs or batch sizes. HF copies prefix KV into independent branches. vLLM can cache repeated question/candidate blocks too: its warm repeated-request measurements **do not isolate media-prefix-only speedup**.

Tiny native-model tests separately verify candidate-hidden-state isolation, final-answer visibility, cleanup after exceptions and unchanged weights, with FP32 and BF16 weights. Audio inputs are rejected for VL; Omni keeps its audio path.

Evidence: [HF cache probe](validation_evidence/qwen3-vl-vl-hf-cache-report.json), [vLLM cache probe](validation_evidence/qwen3-vl-vl-vllm-cache-report.json), [HF synthetic smoke](validation_evidence/qwen3-vl-vl-hf-smoke-report.json), [vLLM synthetic smoke](validation_evidence/qwen3-vl-vl-vllm-smoke-report.json).

## Original-label video pilot

Two videos, containing 10 and 8 original NExT-QA validation questions, were selected by the first two lexicographically ordered video IDs in an existing 200-video subset. All their questions and labels were retained. This is a local-data pilot; the repository does not redistribute its media/annotations or claim an automatic public-download reproduction path for this selection. The independent synthetic smoke workflow is the portable, data-free integration entry point.

| Metric | HF | vLLM |
| --- | ---: | ---: |
| Correct / total | 8 / 18 | 7 / 18 |
| Accuracy | 44.44% | 38.89% |
| NLL | 1.323149 | 1.298714 |
| Multiclass Brier | 0.698866 | 0.695358 |
| ECE, 15 bins | 0.306792 | 0.326210 |
| Two-group wall time, excluding model load | 2.89 s | 15.04 s |
| Model startup | 3.92 s | 33.40 s |

These are **cold, tiny-pilot timings**, including decoding, processing and scoring, without warmup. The first vLLM request also encounters runtime startup work; do not turn these numbers into a general backend speed ranking. HF and vLLM retain different native processing/numerical paths: in this video pilot the HF prompt lengths are two tokens longer. Cross-backend logits, choices and accuracy are therefore **not an equivalence test**. No claim that HF is generally more accurate or that the 4B model beats Omni follows from these samples.

Evidence: [HF video report](validation_evidence/qwen3-vl-vl-hf-nextqa-report.json), [vLLM video report](validation_evidence/qwen3-vl-vl-vllm-nextqa-report.json).

## Omni regression and retained failure

The original six-question Omni HF public workflow was rerun. Every candidate logit matched the previous record exactly (maximum absolute difference **0.0**); original-label accuracy remained **2/6**. This checks the shared adapter changes on that workload, not new Omni vLLM validation. [Regression report](validation_evidence/qwen3-vl-omni-regression-report.json).

The first VL vLLM attempt failed because spawned workers loaded the container's older checkout and did not register `MJevVL`. Engine startup now propagates the current source root and opt-in hooks to workers. That failed run remains separate from the successful outputs and is not included in any score. Raw logs and third-party source annotations remain outside Git.

Experimental HTTP/Tree-KV, audio with VL, other model sizes, quantization and broad accuracy/performance claims are outside this validation.
