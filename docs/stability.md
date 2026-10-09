**English** · [简体中文](stability.zh-CN.md)

Commands outside Docker assume an activated Python 3.11+ virtual environment: create it with `python3 -m venv .venv`, then run `source .venv/bin/activate`. Use `python` for installation and execution in that environment; Docker commands use `python3`. Shell scripts also accept `PYTHON=/path/to/venv/bin/python`. Historical execution records retain their original commands.

# Numerical profiles and multi-question scoring

> Historical results use their recorded configurations. See [current validation](validation_current.md) for later checks and settings that remain unverified.

No model training, weight changes, score adjustment, or generated answer cache
is used. A reference answer is never read by scoring. Stable execution preserves
the intended attention visibility and uses the original LM Head. It changes
floating-point computation, so compare stable batching/cache to stable serial
scoring, not to old native-BF16 logits or accuracy numbers.

## HF

`HFMJevEngine(..., numerics="stable")` is the default. Text computation and MoE
routing use FP32 while the original parameters retain their loaded dtype.
Linear projections use fixed row blocks, and SDPA blocks are aligned to absolute
token positions, so batch size and cached-prefix boundaries do not change the
per-token reduction shapes.
Media uses the official encoders; items are processed independently and exact
repeated items within a forward reuse their features. Native fractional video
positions are retained. There is no vLLM dependency or custom-kernel compilation.

`score_many(..., batch_size=3, use_prefix_cache=True)` and
`score_questions(context, questions, batch_size=3)` perform real three-row
forwards. Each row has independent visibility and a cloned common prefix KV.
The last partial batch can be smaller. This is not zero-copy Tree-KV and does
not merge questions into one conversation. Both single-row and batch scoring
support full or selected-row LM Head
projection; batched questions can have different numbers of labels. Use the
same projection when comparing execution strategies.

```bash
python demo_hf.py --model /path/to/model --input task.json \
  --numerics stable --question-batch-size 3 --prefix-cache --projection full
```

`--numerics native` reproduces the earlier compute path, which can change
answers with batching/cache. It remains available for speed/reference studies.
Stable mode costs additional time and memory; using a cache may amortize prefix
work across questions. Processor and media decoding are separate from scoring.

## vLLM pooling backend

Enable `VLLM_BATCH_INVARIANT=1` before starting the process. This uses vLLM's
existing invariant linear/MoE kernels. The mJev hook also makes residual RMSNorm
invariant and aligns SDPA blocks to absolute token positions so full prefill and
cached suffixes use the same per-token attention shapes. Candidate masks and
the vLLM scheduler remain active; requests are not silently serialized.

```bash
VLLM_BATCH_INVARIANT=1 python demo.py --model /path/to/model \
  --input examples/multiple.json --mode causal
```

Audio/video features are placed by their native typed feature widths and token
IDs. Placement no longer assumes that the entire packed batch is one contiguous
interleaved audio/video region. This fixes text gaps between different requests.
This page applies to `mjev` pooling, not the separate HTTP Tree-KV runtime.

## Reproducible checks

With the pinned dependencies already installed, from the repository root:

```bash
PYTHONPATH=.:runtime MJEV_ENABLE=0 MJEV_ENABLE_PATCHES=0 \
  OMP_NUM_THREADS=4 python -m pytest tests -q -p no:cacheprovider
```

CPU tests check masks, fractional positions, true batches, branch independence,
weight preservation, typed media placement, aligned attention, and failure-safe
hook startup. The release CPU suite passed 102 tests in the existing pinned
environment; clean installation remains unverified.
A passing small consistency test is not dataset accuracy or a production load
claim. Native mode's historical failures remain documented in hf.md.

## Full-weight regression (2026-09-26)

On four 24 GB NVIDIA GPUs, the frozen numerical core passed within-backend comparisons
for 9 media / 27 questions (2 images, 4 audio, 3 audio-video), causal and
isolated masks, one iteration per strategy, full LM Head projection:

| Profile | Serial/batch/cache comparisons | Warm-cache order checks | Maximum logit / probability difference |
| --- | ---: | ---: | ---: |
| HF stable | 216/216 | 54/54 | 0 / 0 |
| vLLM invariant | 270/270 | 54/54 | 0 / 0 |

These counts are repeated comparisons of 27 questions, not dataset accuracy.
References are the same profile's serial scores; cross-backend equality and
universal bitwise determinism are not claimed. The order check replays warmed
questions; it is not an uncached reversed-order test.

Mean causal scoring time for a three-question group was 11.710 s serial without
cache and 3.350 s batched with warm KV on HF; including cold prefix construction,
the latter was 6.816 s. vLLM measured 1.716 s, 0.707 s, and 1.198 s respectively.
These are prepared-input scoring times, excluding media decoding, processor,
startup and networking. Precision/parallelism differ between the two profiles.
HF used real batch=3; vLLM submitted three requests and scheduled at most two
together in the observed steps. Both had 27/27 warm-cache hits per mask; HF
cached suffixes made no media-encoder calls and left the base KV unchanged.

Post-regression release changes add failure-safe hook installation/startup and
honor selected projection in real HF batches. They preserve the tested full-head
computation branch and are checked separately with release CPU and demo tests.

The release `demo_hf.py` was also run on one real audio-video item with three
questions, selected projection, batch=3 and prefix caching; all three reused
523 tokens and no vLLM module was imported. Two-mask public API comparisons
passed 18/18: selected serial/batch/cache scores were exactly equal (12 checks),
while selected/full projection agreed on all six answers with maximum logit
error 1.63e-5 and probability error 3.34e-6. This small release-delta check does
not replace the broader full-projection numerical-core regression above.
