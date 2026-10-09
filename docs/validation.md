**English** · [简体中文](validation.zh-CN.md)

# Validation record — 0.1.0

> Historical results use their recorded configurations. See [current validation](validation_current.md) for later checks and settings that remain unverified.

> Historical release record, not current GPU availability or latest validation status. See [paired evaluation](paired_backends.md) and [stability regression](stability.md).

## Release packaging (2026-09-25)

Validated in a fresh, network-disabled container using the exact base-image digest in the Dockerfile, without access to GPUs:

- Wheel build and installation using `pip install --no-deps --no-build-isolation .`.
- Three CPU tests: mask visibility including the final answer, incompatible batch rejection, and decision tie/nonfinite handling.
- Official local model processor and tokenizer preflight through `demo.py --check-only` on the single- and multi-question examples.
- Docker image build and its actual default entrypoint.
- Python syntax compilation and source-only publication scan.

These checks do not constitute GPU inference. The packaged demo has not been rerun with model weights on GPUs because the reference GPUs are occupied by another service, which was left untouched. The exact model/attention implementation below was tested before packaging; input validation, CLI, output decision metadata and portable paths were added for release. A fresh release GPU regression remains desirable before production use.

## Prior real-model prototype

The saved successful test report used vLLM 0.25.1, PyTorch 2.11.0+cu130 and Transformers 5.13.1 with the official model, BF16 and four 24 GB NVIDIA GPUs. The attention patch and model adapter are unchanged in this release.

| Check | Observed result |
| --- | --- |
| Isolated candidate intervention: other candidate hidden states | max absolute change 0.0 |
| Causal positive control for same intervention | max absolute change 16.375 |
| SDPA causal versus stock Triton logits | max absolute difference 0.3125 |
| Cold versus warm cache logits | max absolute difference 0.375 |
| Prefix cache reuse | cold 0, warm 256 tokens |
| Cross-mask first isolated request | 256 cached tokens, common prefix ends at 265 |
| Repeated isolated request | 368 cached tokens |
| Three concurrent questions | each reused 208 prefix tokens |
| Real inference candidate counts | 2, 3, 5, 7, 30 |

BF16 cache/backend comparisons use a 0.5 raw-logit tolerance, not exact equality. Candidate order tests exercise valid mappings; they do not establish permutation invariance or accuracy improvements. The portable real-model test is `scripts/test_gpu.sh`; its output is ignored by Git.

Raw prior logs, internal paths, model files and private evaluation data are intentionally not distributed. The aggregate observations above are historical results, not newly reproduced release results.
