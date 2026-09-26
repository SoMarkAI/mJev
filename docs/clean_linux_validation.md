**English** · [简体中文](clean_linux_validation.zh-CN.md)

# Clean Linux public workflow — 2026-09-26

The documented HF workflow completed **prepare → real official-model inference → report** on two public MMOU clips and all six original questions. This verifies this installation/configuration, not universal hardware support, a leaderboard score, or reproduction of historical results.

## Verified environment and configuration

- Linux x86_64, Debian 13 container, Python 3.11.16; four 24 GB NVIDIA GPUs, driver 580.105.08.
- PyTorch 2.11.0+cu130, Transformers 5.13.1, Torchvision 0.26.0+cu130, TorchCodec 0.11.0+cpu; system FFmpeg 7.1.5. [Complete environment and checks](validation_evidence/linux-clean-install.json); [all Python dependency versions and source hashes](validation_evidence/linux-public-mmou-hf-report.json).
- Empty venv without system-site packages. Pip downloads and independently verified official weight shards were cached; existing installed packages were not inherited. This was not a fresh network download of all 70.5 GB of weights.
- Official model revision `26291f793822fb6be9555850f06dfe95f2d7e695`. All 15 shard hashes matched pinned Hub LFS metadata: [weight receipts](validation_evidence/linux-official-weight-integrity.json). The runner itself reports Hub provenance; these additional checks were performed separately.
- Code base `04cc2e0a8fd723d678f44b0c69c2e74b4fe94ea4` plus the installation/provenance fixes recorded by the report's actual source hashes. The working tree was dirty; this is not a claim that the unmodified commit passed.
- `configs/public_mini_hf.json`: stable numerics, causal mask, BF16 weights, full LM-head projection, batch size 1, no prefix cache, 4,000-token limit, seed 37, official processor. Decoder explicitly set to `torchcodec`. No weight, attention or candidate-scoring changes were made.

Follow [HF installation](hf.md), including the separate official CPU-codec installation step, then the exact commands in [public reproduction](reproduce.md). The CPU index is for the codec step only; model inference uses CUDA PyTorch. No vLLM installation or custom CUDA compilation was used.

## Actual results

| Check | Outcome |
| --- | --- |
| Clean installation / `pip check` | Passed |
| Seeded CUDA matrix operation | Passed on all four GPUs |
| CPU pytest | 105 passed, 3 optional-vLLM module skips |
| Original media decoding | Both H.264 and AV1 clips decoded without conversion |
| Public prepare / infer / report | 2 clips, 6/6 questions, `_DATA_READY`, `_INFERENCE_COMPLETE`, `_SUCCESS` |
| Full input lengths | 920–1,255 tokens, below 4,000 |
| Image demo `--check-only` | Passed; 119 tokens, A/B/C token IDs 32/33/34; not image-model inference |

Original-label six-question results: **2/6 correct (33.33%)**, NLL **1.346751**, multiclass Brier **0.703255**, ECE **0.327820** (15 bins). These small-sample metrics demonstrate report generation; they do not establish general accuracy or calibration. [Machine-readable report](validation_evidence/linux-public-mmou-hf-report.json).

Model startup took **25.70 s**. The two media groups, containing two and four questions, took **24.63 s** and **16.65 s**: **41.28 s** total. Group timing includes decoding, processing and scoring, with no warmup; model load and Hub resolution are excluded. This is not per-question API latency or a concurrency benchmark.

## Failures retained and fixed

1. Python 3.10 could not install pinned PyAV 18.1.0. The package and guides now require Python 3.11+.
2. An earlier host encountered a PyPI timeout; that installation was not counted as successful.
3. The original HF extra omitted Torchvision, required by the official video processor. It now pins the matching 0.26.0 version.
4. Decord could not decode the selected AV1 clip; its fallback used the removed `torchvision.io.read_video` API. That attempt produced only 2/6 predictions and no successful report. Original samples were retained; the documented run now uses the official TorchCodec reader.
5. The default TorchCodec wheel required a missing CUDA video library. The final clean installation explicitly uses the official CPU wheel, keeping CUDA model inference. Its decoder choice and package version are recorded.

Each failed attempt has separate logs and outputs; none is combined into the final score. Final commands were replayed by a reused reviewer, not a fresh blinded reviewer. Logs and raw outputs are retained outside Git; the public evidence contains configuration, dependency versions, metrics and hashes, without private server paths, media, annotations or weights.

## Remaining scope

One complete HF configuration was tested once. No new vLLM, HTTP/Tree-KV, cache/batch invariance, standalone image full-model, or full benchmark result is claimed. Historical scores retain their original/unverified configuration. See the [current validation index](validation_current.md).
