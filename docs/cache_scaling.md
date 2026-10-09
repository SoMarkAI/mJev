**English** · [简体中文](cache_scaling.zh-CN.md)

# HF prefix KV reuse: controlled performance probe

Model: Qwen/Qwen3-VL-4B-Instruct; revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
Code: `af2215bec742de6d96af4ccb9fb83dd903b4d6c6`, exact archive; experiment harness separately hashed.
GPU: 24 GB NVIDIA GPU; Transformers 5.13.1; PyTorch 2.11.0.
Launcher binds GPU 0 and sets OMP/MKL/OpenBLAS threads to 4. Private operational snapshots are not distributed. No other GPU compute process was observed at checks; no continuous exclusivity guarantee is claimed.
BF16 weights, stable numerics (FP32 text operations), causal attention, full LM-head projection. No model or inference changes.

## Scope and timing
One six-second project-owned animation. Short and long mean actual model input size, controlled through native video sampling and resolution, not video duration. The 16 questions are fixed task prompts, not dataset ground truth; this is not an accuracy evaluation.
Two warmups per cell; five measured repeats, shuffled strategy order per repeat. Strategies execute sequentially; both runs use GPU 0. Synchronized end-to-end score_many time includes media decoding/processing, packing, scoring and fresh prefix prefill. Model load, GC and allocator clearing are excluded. PyTorch allocator is cleared before every measured call, so these values are not directly interchangeable with earlier warm-allocator README figures.
Serial means sequential forwards within one score_many call, which shares decoded media; it is not N independent end-to-end API requests. Baseline batch uses the existing stable implementation, including within-forward vision feature deduplication. Cache batches clone/repeat KV; this is not zero-copy shared storage.
Peak allocated/reserved include model weights and are PyTorch allocator statistics, not whole-device usage.

## Results
Latency is mean ± sample SD in seconds. Memory columns show maximum peak allocated GiB over five repeats.

| Input | Questions | Serial s | Batch s | Cache batch s | Cache vs batch | Serial / batch / cache GiB |
|---|---:|---:|---:|---:|---:|---:|
| short | 1 | 0.173 ± 0.003 | 0.173 ± 0.002 | 0.242 ± 0.012 | 0.71× | 9.76 / 9.76 / 9.81 |
| short | 3 | 0.479 ± 0.010 | 0.358 ± 0.015 | 0.341 ± 0.002 | 1.05× | 9.76 / 9.77 / 9.89 |
| short | 8 | 1.221 ± 0.013 | 0.758 ± 0.005 | 0.582 ± 0.011 | 1.30× | 9.76 / 9.78 / 10.07 |
| short | 16 | 2.430 ± 0.021 | 1.438 ± 0.011 | 0.984 ± 0.017 | 1.46× | 9.76 / 9.80 / 10.32 |
| long | 1 | 2.468 ± 0.012 | 2.467 ± 0.006 | 2.631 ± 0.017 | 0.94× | 9.81 / 9.81 / 11.02 |
| long | 3 | 7.362 ± 0.014 | 6.487 ± 0.032 | 3.118 ± 0.011 | 2.08× | 9.81 / 9.91 / 12.34 |
| long | 8 | 19.535 ± 0.031 | 16.464 ± 0.026 | 4.246 ± 0.020 | 3.88× | 9.81 / 12.09 / 15.49 |
| long | 16 | 38.986 ± 0.022 | 32.620 ± 0.048 | 6.287 ± 0.014 | 5.19× | 9.81 / 15.80 / 20.60 |

A ratio above 1 means the cached strategy is faster.

## Input lengths and numerical checks
- short: full prompt 100–117 tokens; cached prefix [63]; max logit delta against same-input serial reference 0.0.
- long: full prompt 2303–2320 tokens; cached prefix [2266]; max logit delta against same-input serial reference 0.0.

All reported successful runs passed finite-score, within-candidate probability sum and same-input decision checks. Different sampling profiles are not required to produce equal answers.

## Limits
No general threshold, accuracy improvement, native BF16 speed, vLLM speed, Qwen3-Omni-30B speed or audio performance is established. The initial intermediate 178-token-prefix run is retained in results/ but excluded from the short/long table; the corrected long configuration and harness are stored separately. Use raw timings rather than rounding to decide near ties.

Raw evidence: [initial run](validation_evidence/hf-cache-scaling/results/raw.json), [corrected long run](validation_evidence/hf-cache-scaling/results-long/raw.json), [combined summary](validation_evidence/hf-cache-scaling/combined-summary.json). Each run directory includes configuration, dependency/source provenance and loading diagnostics. The intermediate 178-token case is retained transparently, not selected for the main table. The CPU checker below recomputes the aggregates and compares the raw candidate scores. It checks the archived records without running GPU inference.

## Check the published evidence (CPU only)

From the repository root, run:

```bash
python docs/validation_evidence/hf-cache-scaling/analyze.py
```

This standard-library checker validates the frozen raw results, script hashes, cross-run settings, candidate mapping, normalization and same-input scores, then recomputes all eight rows. It does not run a model or modify evidence. Use `--out outputs/cache-scaling-check` to export the regenerated English report and summary. Completion markers alone do not establish correctness; the checker reads the actual scores.

## Run a new GPU measurement

Install the [HF video environment](hf.md) first. The recorded core commit `af2215bec742de6d96af4ccb9fb83dd903b4d6c6` is absent from the public Git history. The commands below archive your current commit and record it for a new measurement. They use the public motion fixture and write to a separate output directory.

The archived harnesses are unchanged: `run-initial.py` includes the intermediate 178-token case, and `run.py` measures the long input. Their provenance strings describe the historical run; verify model integrity separately for a new run. These commands reproduce the measurement procedure with the selected code commit, rather than the unavailable historical source archive.

```bash
BENCHMARK_DIR="$PWD/docs/validation_evidence/hf-cache-scaling"
CORE_COMMIT="$(git rev-parse HEAD)"
CORE_DIR="$(mktemp -d)"
RUN_DIR="$(mktemp -d)"
git archive "$CORE_COMMIT" | tar -x -C "$CORE_DIR"
MODEL_DIR="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3-VL-4B-Instruct", revision="ebb281ec70b05090aa6165b016eac8ec08e71b17"))')"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONPATH="$CORE_DIR"
python "$BENCHMARK_DIR/run-initial.py" --model "$MODEL_DIR" \
  --source "$CORE_DIR/examples/motion-demo/motion.mp4" --out "$RUN_DIR/results" \
  --commit "$CORE_COMMIT" --repeats 5
python "$BENCHMARK_DIR/run.py" --model "$MODEL_DIR" \
  --source "$CORE_DIR/examples/motion-demo/motion.mp4" --out "$RUN_DIR/results-long" \
  --commit "$CORE_COMMIT" --repeats 5
```

Each harness writes raw scores/timings, its environment, per-cell summary and completion marker under `RUN_DIR`. Do not overwrite the frozen repository evidence with a replay. The CPU checker above is for the published five-repeat artifact, not an arbitrary new run. Select a free GPU; the recorded long/16 cache case peaked at 20.60 GiB allocated and 23.11 GiB reserved, excluding non-PyTorch GPU overhead. Smaller cards may report OOM. A completion marker means all cells were attempted; inspect cell status.

The archived measurements predate this documentation update. The CPU evidence checker and script help were checked again; the new GPU command sequence has not been executed.

## Earlier three-question fixture

This earlier measurement uses the [minimal rectangle example](../examples/video-demo/input.json), with a different input and timing regime from the scaling study above. It is retained as a separate historical result.

| HF execution | Three-question wall time (mean of two runs) |
| --- | ---: |
| Serial, no KV reuse | 0.471 s |
| Batch of three, no KV reuse | 0.301 s |
| Batch of three, prefix KV reuse | 0.326 s |

Conditions: one 24 GB NVIDIA GPU, Qwen3-VL-4B, `causal`, `stable`, BF16 weights and full projection; two repetitions after one warmup. Includes media decoding, processing and scoring, plus fresh prefix prefill for cached calls; excludes model loading. Plain batching was faster on this short input. Cache benefits depend on the workload; these observations do not establish production throughput or a general speedup factor. [Per-run timings](validation_evidence/qwen3-vl-vl-hf-cache-report.json) · [Full validation](qwen3_vl_validation.md)
