**English** · [简体中文](reliability_review.zh-CN.md)

# Reliability fixes — 2026-09-26

This update fixes interface, reporting and experimental-runtime boundaries. It does not change model weights, attention equations or score calculations. Existing explicitly configured experiments retain their original records.

## Behavior changes

- Main HF/vLLM CLI and scoring interfaces now default to `causal`. Callers relying on the old vLLM default must pass `isolated` explicitly to retain that attention mode.
- Paired reports use the frozen manifest for every correctness statistic and reference category. Conflicting result-row labels, dataset, modality or reference type cause rejection. Manifest hashes and exact per-mode coverage are recorded/checked.
- In-place report and media-audit attempts revoke their old completion marker before validation. Final reports are published before `_SUCCESS`. Inference tools that require a new output directory continue to reject reuse instead of modifying completed experiments.
- The model validation probe forwards sampling options for both `video` and `audio_video`; tests cover the probe boundary and the Omni media-processor call.
- CLI preflight and inference use shared structural validation with scoring APIs. Empty question lists and misordered candidate dictionaries fail before model loading. `demo_hf.py` separates parsing, preflight, scoring and output.
- Experimental HTTP media access uses explicit local roots, bounded file/data-URL reads and opt-in HTTPS hosts. Validated bytes are passed to vLLM. Runtime hook failures abort startup; supported-version and hook-readiness checks are enforced. The duplicate route was removed. See [runtime configuration](integrated.md#media-access-and-startup-checks).
- Grouped evaluation saves each completed media group atomically, alongside cumulative predictions and progress. Later failures preserve completed work. This is not automatic resume.

## Checks

- Local Python 3.11: **175 passed**, four optional-vLLM modules skipped.
- Pinned Linux vLLM 0.25.1 environment: **184 passed** CPU tests, including API integration.
- Actual experimental-runtime hook installation on that pinned version: all five required hook flags ready.
- Failure injections cover manifest/row reference conflicts, missing rows after successful reporting, media traversal/symlinks/size bounds, private remote addresses/redirects, partial hook installation, and later-group scoring failures.

These are software regression checks. No new full-weight GPU inference, HTTP Tree-KV parity result or performance ranking is claimed by this update. Existing GPU records describe their own source snapshots.

## Evidence still to build

Question-count/input-size latency and peak-memory sweeps, and original-label task evaluations with media-removal and candidate-order controls, are worthwhile new experiments. They are not code fixes or evidence already obtained. Comparisons with other projects require source-specific review; this change only clarifies the distinction between independent question branches and candidate isolation within a question. Candidate isolation remains an optional structure for controlled experiments, not a demonstrated higher accuracy ceiling.
