**English** · [简体中文](benchmark_validation.zh-CN.md)

# AV mini benchmark validation — 2026-09-25

> See the [current validation index](validation_current.md) for backend boundaries and latest checks. Historical results retain their recorded configuration; missing model revisions, complete dependencies or run settings are unverified, never inferred from current defaults.

> Historical data audit only. Subsequent model results are documented in [paired evaluation](paired_backends.md).

This records data/tool validation, **not model evaluation**. No GPU inference, training or annotation generation was performed.

| Dataset | Questions | Distinct media | Media bytes |
| --- | ---: | ---: | ---: |
| MMAU-test-mini | 1000 | 1000 audio files | 1,815,779,127 |
| Video-MME-v2 | 1000 | 250 videos | 31,524,539,961 |
| MMOU Test Mini | 1000 | 903 videos | 21,749,076,990 |
| Total | 3000 | 2153 | 55,089,396,078 |

Source annotation revisions and deterministic selection are defined in `prepare.py` and recorded in the external bundle's `selection.json`. The frozen manifest SHA256 is:

```text
007914633492397a6853cf33c4b07b50b330b5d8d23193513f716ea33945e884
```

Verified:

- All 3000 questions, ordered options and original answers reconstruct exactly from source annotations; references also match the separately downloaded annotation files.
- All 2153 selected files exist, with no unexpected media. Download receipts match file sizes and SHA256. Selected ZIP members passed CRC checks. MMOU videos and the MMAU source Parquet additionally match upstream pinned LFS SHA256 values.
- ffprobe found valid streams and positive duration in every media file. Every selected video has an audio stream. The audio containers are 954 WAV and 46 MP3, preserved byte-for-byte under `.audio` filenames.
- Candidate counts remain variable: MMAU has 2/4/5/8 choices; Video-MME-v2 has 2/3/5/6/7/8; MMOU has 10.
- The metric implementation matches analytical uniform-distribution NLL/Brier values across the real 3000-record manifest. This is a synthetic scoring check, not a model result.
- CPU tests cover dynamic labels, numerical stability, ECE, ties, missing/invalid predictions, input redaction, failure-stop/resume behavior, selective ZIP extraction and range refusal, source formatting, processor FPS/patch size, and duration warnings.
- Official Qwen3-Omni processor preflight succeeds on one real WAV, one real MP3 and one real audio-video example, with contextual label token checks. The video preflight uses four frames and an explicit small pixel budget only as a smoke test; it is not a benchmark inference policy. It produces video time-grid metadata and audio features without loading model weights.

The processor preflight uses Transformers 5.13.1, qwen-omni-utils 0.0.9, librosa 0.11.0 and audioread 3.0.1. Newer librosa 1.0 is not used with this utility version. Empty FPS lists are omitted; a single video's sampled FPS is forwarded as a scalar compatible with this Transformers version. Patch size comes from the actual official processor.

Source caveats retained in the bundle:

- Nine MMOU questions whose videos are absent from the pinned media index were excluded **before** deterministic sampling; the eligible pool is 4991 questions.
- Eight selected MMOU videos (affecting 11 selected questions) have file durations differing from annotation metadata by more than two seconds; the largest absolute difference is 44.739 seconds. Their bytes match upstream LFS objects. These are reported in `duration_warnings.json`; no original fields, labels or videos were silently replaced. Use actual decoded timing, and review these entries before making time-sensitive claims.

The audit checks file integrity and container/stream metadata, not a complete frame-by-frame decode or human correctness of each source answer. `_SUCCESS` in the external bundle certifies this data audit only. The experimental AV adaptation is documented separately in [the pilot guide](av_pilot.md). Its 10-question CPU processor preflight passed (154–3027 tokens); GPU scoring, mask isolation and cache assertions have not yet been executed for AV. This data audit does not certify model inference.
