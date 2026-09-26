**English** · [简体中文](benchmark.zh-CN.md)

Commands outside Docker assume an activated Python 3.11+ virtual environment: create it with `python3 -m venv .venv`, then run `source .venv/bin/activate`. Use `python` for installation and execution in that environment; Docker commands use `python3`. Shell scripts also accept `PYTHON=/path/to/venv/bin/python`. Historical execution records retain their original commands.

# Small audio/video benchmark

> The image-only runtime boundary below describes the initial release. For subsequent audio/video support and full-weight validation, see [HF](hf.md) and [paired evaluation](paired_backends.md).

The `mjev.benchmark` package builds a frozen, original-label evaluation suite. It does not train a model or regenerate annotations. Media and annotations stay outside Git. Dataset licenses remain separate from the mJev code license.

| Source | Questions | Selection unit | Original candidate counts |
| --- | ---: | --- | --- |
| [MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) | 1000 | All mini examples | Preserved, including non-four-choice questions |
| [Video-MME-v2](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2) | 1000 | 250 videos, all four associated questions per video | Preserved |
| [MMOU Test Mini](https://huggingface.co/datasets/nvidia/MMOU) | 1000 | Questions with media listed in the pinned Test Mini video index | 10 in the current source |

Question selection ranks `SHA256(seed:dataset:source_id)`, with seed `20260925`. Selection is independent of reference labels and network timing. Reordering source rows does not change the sampled question/video IDs. Video-MME-v2 retains complete question groups; the common evaluator below reports **per-question** metrics, not its official group-based leaderboard score. These sampled scores must not be described as full-benchmark scores.

MMOU Test Mini's pinned annotations contain 5000 questions, but nine reference videos are absent from the pinned `test/` media index. The builder logs all nine excluded question IDs and samples from the remaining 4991 questions. The default selection needs 903 distinct MMOU videos. Unavailable videos are not silently replaced after download failures.

## Prepare and download

The data tools do not import torch or vLLM. From a checkout, use Python 3.11+ and install their lightweight dependencies:

```bash
python -m pip install 'requests>=2.31' 'pyarrow>=15'
export BENCH_ROOT="$HOME/datasets/mjev_av_mini_v1"
python -m mjev.benchmark.prepare --output "$BENCH_ROOT"
python -m mjev.benchmark.fetch --root "$BENCH_ROOT" --workers 8
```

With FFmpeg installed, `BENCH_ROOT=/path/to/data bash scripts/build_av_benchmark.sh` runs preparation, download and audit sequentially; any failure stops the script.

`prepare` downloads annotations and metadata only. Its revisions are pinned in `prepare.py`; `selection.json` records IDs through the manifest, sampling method, exclusions and hashes. Reusing a directory with a different seed or requested size is rejected. Use a new directory for a new suite.

`fetch` supports `--dataset mmau|video_mme_v2|mmou` and `--limit N` (download jobs, not questions) for a pilot. It downloads exactly the frozen plan:

- MMAU: the official mini Parquet contains the selected 1000 audio entries. Extract its original bytes without decoding/re-encoding. Because actual containers include WAV and MP3, files use a neutral `.audio` suffix; decode by content. The original question and option strings, including label prefixes, remain in `original`.
- Video-MME-v2: HTTP Range reads ZIP directory/header metadata and selected MP4 members. It never falls back to downloading entire ZIPs. ZIP CRC and member sizes are verified. Small ZIP framing reads can include adjacent bytes; no unselected video is extracted or retained.
- MMOU: fetch only selected `test/<video_id>.mp4` files from the [media repository](https://huggingface.co/datasets/sonalkum/MMOU-Videos). Do not download the full corpus or captions.

Only one downloader should own a given file/job at a time. Completed downloads have SHA256 receipts. Failed direct downloads retain `.part` files for explicit resume; interrupted ZIP members restart that selected member only. There are no automatic retries: an error stops new submissions, allows already in-flight jobs to finish and records a paused status. Rerun the same command to explicitly resume.

An optional HTTPS mirror can be selected via `MJEV_HF_ENDPOINT`; this changes transport only, not the recorded official source URLs, revisions or sample IDs. Do not put proxies, tokens or private server addresses in repository configuration.

Media duration varies substantially. A thousand video questions can still require many gigabytes; inspect free disk space before starting. No frame/audio truncation is performed by the dataset builder.

## Layout and record schema

```text
$BENCH_ROOT/
├── manifest.jsonl          # 3000 records, paths relative to this directory
├── mmau.jsonl             # 1000 records
├── video_mme_v2.jsonl      # 1000 records
├── mmou.jsonl              # 1000 records
├── selection.json          # sampling, revisions, counts, annotation hashes
├── download_plan.json
├── download_status*.json
├── sources/                # source annotations and mini audio Parquet
├── media/{mmau,video_mme_v2,mmou}/
├── media_index.json        # produced by audit: hashes, streams, durations
├── validation.json
└── _SUCCESS                # only after final data audit passes
```

Illustrative record (not copied from a benchmark question):

```json
{
  "id": "mmou:original-question-id",
  "dataset": "mmou",
  "source": {"repository": "nvidia/MMOU", "revision": "pinned-commit", "file": "MMOU_TEST_MINI.json", "original_id": "original-question-id"},
  "split": "test-mini",
  "task": ["Temporal Understanding"],
  "modality": "audio_video",
  "media_path": "media/mmou/example.mp4",
  "media": {"type": "video", "path": "media/mmou/example.mp4"},
  "question": "Original question text",
  "candidates": {"A": "Original first option", "B": "Original second option"},
  "label": "B",
  "metadata": {},
  "original": {}
}
```

`candidates` is an insertion-ordered mapping. Candidate order, text, count and reference answer are retained. Only explicit option label delimiters such as `A. ` or `(A) ` are separated from the answer text; the exact original form is stored in `original` and audited by reconstruction. There is no fuzzy answer matching or relabeling. Full-video inputs retain original evidence timestamps as metadata, but these are **never automatically passed to inference or used for cropping**.

## Loader and inference interfaces

```python
from mjev.benchmark.dataset import MJevDataset

suite = MJevDataset("/path/to/benchmark/manifest.jsonl")
for record in suite:
    payload = suite.model_input(record)
    messages = suite.qwen_messages(record)
```

For a Qwen environment with `qwen-omni-utils[decord]==0.0.9`, `librosa==0.11.0` and `audioread==3.0.1` installed, the optional adapter also constructs the official vLLM multimodal input and checks contextual label tokens:

```python
from transformers import AutoProcessor
from mjev.benchmark.adapters import qwen_vllm_input

processor = AutoProcessor.from_pretrained("/path/to/official/model", local_files_only=True)
record = suite.records[0]
prompt, label_token_ids = qwen_vllm_input(
    suite, record, processor,
    use_audio_in_video=(record["modality"] == "audio_video"),
    video_options=None,  # set and record an explicit policy for video runs
)
# Pass prompt to an AV-capable scoring backend; see av_pilot.md for the experimental mJev path.
```

This follows the [official Qwen audio/video processor contract](https://github.com/QwenLM/Qwen3-Omni#use-audio-in-video). `use_audio_in_video` must agree between preprocessing and backend processor configuration. The adapter reads patch size from the official processor and forwards the utility's actual sampled FPS, so temporal metadata is not silently replaced by a default FPS. `label_token_ids` refer to the next-token position after the exact `Answer:\n` prefill.

`model_input` exposes only `id`, `modality`, resolved `media_path`, `question` and `candidates`. It omits labels, original annotations, task metadata and evidence locations. `qwen_messages` supplies an audio/video content item plus the original question and choices for the official Qwen processor. Media decoding remains the backend's responsibility. For audiovisual evaluation use the official Qwen audio-in-video path, handle genuinely silent videos explicitly, and record any frame sampling, resizing or audio truncation policy in the inference run configuration. The loader does not append subtitles or judge captions.

Implement a scorer accepting that payload and returning exactly one label-to-score mapping:

```python
async def score(payload):
    # Run your AV-capable backend here; do not read labels from the manifest.
    result = await backend.score(payload)
    return {"logits": result.raw_candidate_logits}
```

Then run resumably:

```bash
python -m mjev.benchmark.run \
  --manifest "$BENCH_ROOT/manifest.jsonl" \
  --scorer your_backend:score --run-config configs/your_av_run.json \
  --concurrency 3 --output outputs/av_predictions.jsonl
```

Supply your own run config JSON identifying model revision, mode, precision and media preprocessing (frames/fps/pixels, audio handling and any truncation). A `.run.json` sidecar binds predictions to that configuration, scorer and manifest hash; resume rejects changed settings. The runner records this configuration but the backend must actually apply it. Use a new output path for each model/configuration.

Each output record must contain `id` and either `logits` or `probabilities`, keyed by **all** original labels. `adapters.from_mjev_result` converts mJev-style candidate results after checking text/order. `adapters.from_qwen_logits` selects label logits from the raw next-token vocabulary vector and verifies single-token labels against the exact rendered template/prefill. Neither adapter calls `generate()` or changes model weights.

**Runtime boundary (historical):** the initial image-only release did not certify AV inference. Current HF and pooling paths support AV; see [paired evaluation](paired_backends.md) and [stable regression](stability.md). Dataset/tool checks alone still imply no model accuracy.

## Evaluate

```bash
python -m mjev.benchmark.evaluate \
  --manifest "$BENCH_ROOT/manifest.jsonl" \
  --predictions outputs/av_predictions.jsonl \
  --bins 15 --output outputs/av_metrics.json
```

- **Accuracy:** mean argmax correctness; exact ties use original candidate order.
- **NLL:** mean negative natural logarithm of reference-label probability. Logits use stable log-sum-exp. A true zero target probability gives infinity, serialized as `null` with `nll_is_infinite=true`; it is not secretly clipped.
- **Brier Score:** mean sum across candidates of squared difference from one-hot ground truth, without dividing by candidate count (range 0–2).
- **ECE:** top-label confidence calibration, 15 equal-width bins by default. Bins are left-closed/right-open; the last includes 1. ECE depends on binning and sample size.

Probabilities must be finite, in [0,1], and sum to one within 1e-6. They are not silently renormalized. Hard answers alone are insufficient for NLL/Brier/ECE. Unknown/duplicate IDs, label mismatches and missing predictions fail by default. `--allow-partial` explicitly reports coverage and missing IDs, with metrics restricted to the scored subset. Reports include dataset, modality, task and candidate-count breakdowns. Multi-task examples occur in multiple task groups; do not add those group counts as if disjoint.

## Audit and tests

Install `ffprobe` (from FFmpeg), then run:

```bash
python -m mjev.benchmark.audit --root "$BENCH_ROOT"
python -m pytest tests/test_benchmark.py -q
```

The audit reconstructs every original question/options/answer, checks the frozen manifest hash, validates file size/SHA256 receipts and inspects media streams and duration with ffprobe. It detects missing/unexpected media and records videos without audio. `_SUCCESS` means the data audit passed, not that inference ran. This is container/stream inspection, not a complete decode of every audio sample or video frame.

The initial bundle's exact counts, checks and source caveats are recorded in
[benchmark_validation.md](benchmark_validation.md). Duration discrepancies remain
in a separate `duration_warnings.json`; they do not overwrite original metadata.
