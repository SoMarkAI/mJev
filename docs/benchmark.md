# General evaluation

[简体中文](benchmark.zh-CN.md) · [Document evaluation](docjev/evaluation.md) · [HF inference](hf.md)

Use `docjev.evaluate` for document checkpoints. For general image, video and audio tasks, supply a frozen JSONL manifest. Dataset downloading and training are maintained separately.

## Input

One question per line, with media paths relative to the manifest. Preserve original questions, candidate order and reference labels:

```json
{"id":"q1","dataset":"my-data","source":{"revision":"my-frozen-version"},"task":"color","modality":"image","media_path":"page.png","media":{"type":"image","path":"page.png"},"question":"What color is the rectangle?","candidates":{"A":"Red","B":"Blue"},"label":"A"}
```

Modalities are `image`, `video`, `audio` and `audio_video`. Qwen3-VL accepts images and video without audio; use Qwen3-Omni for audio tasks. Reference labels, original annotations and evidence timestamps never enter the model input or determine media crops.

## Run

From the checkout with the HF dependencies installed:

```bash
python -m mjev.benchmark.grouped --manifest data/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/evaluation
```

The configuration pins the official model ID, 40-character revision, numerics, projection, cache, batch size and media settings. The runner downloads or reuses that Hub snapshot. For vLLM use `configs/qwen3_vl_4b_vllm.json` and the pinned vLLM environment. For local document checkpoints, use [document evaluation](docjev/evaluation.md).

Use a fresh output directory. `run.json` records source, model configuration and media hashes. `predictions.jsonl` contains candidate logits; `report.json` contains metrics. Completed groups survive later failures in `groups/` and `progress.json`. Failure writes `ERROR.json`; only complete coverage writes `_SUCCESS`. The grouped runner does not resume automatically.

Recompute metrics offline:

```bash
python -m mjev.benchmark.evaluate --manifest data/manifest.jsonl \
  --predictions outputs/evaluation/predictions.jsonl --output outputs/metrics.json
```

For an existing custom scorer, see `python -m mjev.benchmark.run --help`. This runner supports resumption bound to the manifest and configuration. `MJevDataset.model_input()` restricts the fields sent to the scorer.

## Metrics and smoke checks

Reports include Accuracy, NLL, Brier Score and ECE. Unknown/duplicate IDs, label mismatches and missing predictions fail by default. Probabilities are never silently renormalized; ties follow input candidate order. Report only the evaluated sample scope.

Synthetic integration checks require FFmpeg:

```bash
python -m mjev.benchmark.visual_smoke --root data/visual-smoke
python -m mjev.benchmark.grouped --manifest data/visual-smoke/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/visual-smoke
```

The generated rectangle image and static video are owned fixtures, not natural-data benchmark accuracy. [Cache comparisons](cache_scaling.md) separately check serial, batch and prefix reuse.
