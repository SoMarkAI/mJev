**English** · [简体中文](paired_backends.zh-CN.md)

Commands outside Docker assume an activated Python 3.11+ virtual environment: create it with `python3 -m venv .venv`, then run `source .venv/bin/activate`. Use `python` for installation and execution in that environment; Docker commands use `python3`. Shell scripts also accept `PYTHON=/path/to/venv/bin/python`. Historical execution records retain their original commands.

# Paired HF / vLLM evaluation

> See the [current validation index](validation_current.md) for backend boundaries and latest checks. Historical results retain their recorded configuration; missing model revisions, complete dependencies or run settings are unverified, never inferred from current defaults.

Run the official local Qwen3-Omni Thinker through HF and the main package's vLLM
pooling backend, not the runtime HTTP server. No generation, trained weights or
Decision Head is introduced. Both causal and candidate-isolated modes are tested.

Use a Linux environment containing the pinned HF and optional vLLM dependencies.
The inspected preinstalled image is recorded in the local compute ledger; this is
not evidence that arbitrary clean installs have been validated.

```bash
PYTHONPATH=. python benchmarks/integrated/compare_backends.py --freeze --root /path/to/av-benchmark --infinity /path/to/infinity-eval --out outputs/paired
VLLM_BATCH_INVARIANT=0 MJEV_ENABLE=1 VLLM_USE_V2_MODEL_RUNNER=0 PYTHONPATH=. python benchmarks/integrated/compare_backends.py --backend vllm --model /path/to/model --out outputs/paired --numerics native --model-revision "$MODEL_REVISION"
MJEV_ENABLE=0 PYTHONPATH=. python benchmarks/integrated/compare_backends.py --backend hf --model /path/to/model --out outputs/paired --numerics native --model-revision "$MODEL_REVISION"
PYTHONPATH=. python benchmarks/integrated/report_comparison.py --out outputs/paired
```

The GPU steps run sequentially and each requires four free GPUs. Use an isolated
output directory; neither runner overwrites an existing backend result directory.
For a smoke run, copy the frozen manifest to another output directory and add
`--smoke` to each backend invocation.

Inputs and references are frozen before either backend runs. File hashes are
rechecked at runtime. Media is decoded with the same qwen-omni-utils path, with
image/video resizing done once, explicit video frame/pixel settings, and audio
right-zero-padded to the feature extractor hop size for both backends. The
processor uses full audio and sampled frames; reference answers are never model
inputs. Expanded prompt token hashes must match the common HF processor output.
This does not independently hash every internal vLLM encoder feature tensor.

Both backends disable prefix KV reuse. vLLM also disables multimodal processor
caching. HF uses the unchanged full LM head at the last position, then selects
single-token candidate labels. vLLM uses its original LM head/pooling readout.
HF distributes layers over four GPUs; vLLM uses TP4. Kernel differences remain.

Timing is serial, concurrency one: question template/processor through completed
scoring, after CUDA synchronization. Common media decode and model startup are
reported separately and excluded from per-question scoring latency. The runner
also performs untimed common preprocessing for verification. Whole evaluation
wall time includes verification, decoding, warmups and bookkeeping; it is not
production serving throughput. Per-modality/mode warmups are excluded from score
samples. One timed observation per question is recorded; P95 is across questions,
not a repeated fixed-request or concurrent-load P95.

Infinity labels are generated and not human reviewed: report reference agreement,
not audited accuracy. AV uses original MCQ labels and the existing unanimous
Clotho binary adapter. Freeform/disagreeing Clotho references are explicitly
excluded. Any common decode/processor rejection remains in rejected.json, and
paired reports require identical accepted ID coverage in both backends.

Raw logits, probabilities, decisions, durations, token hashes and references are
saved per question/mode. The report includes original-label Accuracy (or generated
reference agreement), NLL, Brier, ECE, mean/median/P95 latency, serial scoring rate,
reference ties and paired answer changes. Neither labels nor media are published
with the Apache-2.0 code; their original licenses remain applicable.

## Measured uncached evaluation (2026-09-26)

Four 24 GB NVIDIA GPUs, official BF16 Thinker, serial requests, one timed observation
per question/mode. Both modes in the vLLM pooling backend use the mJev custom dense
SDPA attention path, including causal; this is not a stock-vLLM performance test.
HF uses layer sharding and vLLM uses TP4. No weights were trained or changed.

| Data | Questions | Mode | vLLM metric | HF metric | vLLM mean s/question | HF mean s/question |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Audio | 526 | causal | 81.37% | 81.37% | 0.129 | 0.309 |
| Audio | 526 | isolated | 76.81% | 78.33% | 0.129 | 0.315 |
| Video with audio | 423 | causal | 51.77% | 52.25% | 0.929 | 0.819 |
| Video with audio | 423 | isolated | 11.11% | 10.64% | 0.932 | 1.031 |

Infinity is generated-reference agreement, not human-GT accuracy. Audio combines
226 unanimous-reference Clotho-AQA binary questions and 300 MuChoMusic questions.
Video combines 411 MMOU and 12 Video-MME-v2 questions. Scores use original dataset
labels for AV and are not official leaderboard aggregates.

Of 1,335 source AV questions, 374 freeform/disagreeing Clotho questions were
excluded before freezing. The frozen set contained 961 AV and 300 image questions.
Both backends then rejected the same 12 questions: 10 exceeded the 4,000-token
full-input limit, and one video decode failure affected two questions (torchcodec
frame exhaustion followed by unavailable torchvision read_video fallback). Thus
1,249 questions were paired, 2,498 scoring rows per backend, 4,996 total.

Independent recomputation verified coverage, input-token hashes, decisions,
softmax, metric aggregates, and zero cached tokens. This does not prove identical
encoder feature tensors or every checkpoint shard. Scope/provenance warnings
remain: generated image references, filtered subsets, no repeated runs, and no
concurrent throughput measurement. Private raw outputs and third-party media are
not distributed. Current results favor causal over candidate isolation; they do
not establish why isolation degrades performance.

> Set MODEL_REVISION to the actual 40-character commit of your local model. The native commands are an explicit rerun profile, not a reconstruction of missing historical settings or a guarantee of the old scores. For a publicly downloadable workflow, use [public mini evaluation](reproduce.md).
