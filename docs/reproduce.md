**English** · [简体中文](reproduce.zh-CN.md)

# Public small evaluation: prepare → infer → report

This is a new, deliberately small **MMOU short-video subset**, not a reproduction of the historical 1,249-question private paired set or an official leaderboard score. No internal bundle, Infinity file or server path is needed. The default selects two videos and retains all their original questions (currently six). Original options and answers are preserved. Selection uses a fixed seed and annotation duration, never reference labels. A short duration is not a token-length guarantee: overlength/decoding failures stop inference explicitly.

## Installation and fixed configuration

Use the Linux HF installation in [hf.md](hf.md), FFmpeg and `pip install -e '.[hf,test]'`. vLLM additionally requires the pinned optional backend and its [deployment environment](vllm.md). All commands run from the checkout. See [actual installation checks](installation_validation.md) before interpreting these instructions as validated on your platform.

`configs/public_mini_hf.json` and `configs/public_mini_vllm.json` specify numerics, mode, BF16 weight dtype, projection, cache policy, batch size, full-input limit, seed, context and video preprocessing. The official model is pinned to `26291f793822fb6be9555850f06dfe95f2d7e695`, resolved from the public model API for this new workflow. This is **not evidence of the revision used by historical experiments**. `infer` uses `snapshot_download` with this exact revision, reusing its Hub cache if present; it may download approximately 70.5 GB of model files. It does not accept a floating branch/tag or silently use an unrelated local model directory.

The initial profiles use causal, stable numerics, full projection, one question at a time, and no prefix caching. HF can explicitly enable prefix caching/batching in a copied config; vLLM mini currently rejects those options to keep its serial baseline unambiguous. This does not remove existing backend caching capabilities.

Install the CPU codec wheel before the HF extra, exactly as in the HF guide. `torchcodec==0.11.0+cpu` comes from the official PyTorch CPU index, not default PyPI; the subsequent HF install uses default PyPI for CUDA PyTorch.

## 1. Download only selected public media

```bash
python -m mjev.benchmark.public_mini prepare \
  --root data/public-mmou-mini --videos 2 --max-seconds 15 --seed 20260926
```

Use a **new directory**. The command downloads pinned MMOU Test Mini annotations and the pinned media index, ranks eligible video IDs by SHA256, retains all original questions for selected videos and downloads only those files. It records exclusions, source revisions, annotation/manifest hashes and media receipts. `_DATA_READY` means byte checks passed, not full decoding or inference. Errors produce `ERROR.json`; there is no silent replacement or automatic retry. After a failed preparation, inspect it and choose a new directory for an explicit retry.

Annotations retain their upstream Apache-2.0 declaration; the original video/audio rights remain unverified. Media are not included in Git and are not relicensed. See [third-party terms](../THIRD_PARTY_LICENSES.md).

## 2. Run official-model inference

HF:

```bash
FORCE_QWENVL_VIDEO_READER=torchcodec MJEV_ENABLE=0 MJEV_ENABLE_PATCHES=0 \
python -m mjev.benchmark.public_mini infer \
  --root data/public-mmou-mini --config configs/public_mini_hf.json \
  --out outputs/public-mmou-hf
```

Optional vLLM pooling, in its supported four-GPU environment:

```bash
MJEV_ENABLE=0 MJEV_ENABLE_PATCHES=0 \
python -m mjev.benchmark.public_mini infer \
  --root data/public-mmou-mini --config configs/public_mini_vllm.json \
  --out outputs/public-mmou-vllm
```

The runner enables vLLM hooks only for that backend before engine construction. Use a new output directory for each run. Original labels and evidence timestamps are excluded from scoring payloads. Inference failures retain completed predictions and `ERROR.json`; no successful report is generated for partial runs. This runner does not resume partial inference or silently skip questions.

`run.json` records the code commit **and actual source-file hashes**, dirty-tree state, immutable model revision and resolution status, processor/config hashes, every installed dependency version, Python/platform, GPU names, engine settings and requested inference configuration. Source hashes distinguish local uncommitted changes from the recorded base commit. Hub snapshot resolution is not an independent checksum of every weight shard. Existing local-bundle runners label their caller-declared model revision `declared_local_unverified`.

Timing is per media group and includes decode, processor and scoring. Model load is reported separately; there is no warmup. It is not per-question API latency or steady-state throughput. HF placement and vLLM TP4 differ, so this is not a matched-kernel speed claim.

## 3. Generate metrics and report

```bash
python -m mjev.benchmark.public_mini report --out outputs/public-mmou-hf
# After running the optional backend:
python -m mjev.benchmark.public_mini report --out outputs/public-mmou-vllm
```

Produces `report.json`, `REPORT.md` and `_SUCCESS` only after complete coverage and manifest checks. Metrics are Accuracy, NLL, Brier and ECE from the shared evaluator. Raw logits and grouped timings remain in `raw.jsonl`; `predictions.jsonl` contains label-to-logit predictions. Preserve the full output directory and data selection metadata when sharing results, subject to source licenses; do not publish private paths or restricted media.

## Historical experiments

The legacy `benchmarks/integrated/` runners remain available for their original external bundles. New runs require explicit `--numerics` and a 40-character `--model-revision`; `evaluate_hf.py` also requires projection and question batch size. They write run evidence. vLLM must be launched with `VLLM_BATCH_INVARIANT=1` for stable or `0` for native. Historical artifacts without configuration evidence are reported as **unverified**, never assigned today's defaults. The old reported scores are unchanged; see [current validation index](validation_current.md).
