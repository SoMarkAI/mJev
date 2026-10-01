**English** · [简体中文](hf.zh-CN.md)

[Qwen3-VL-4B uses the same HF API; see the model guide for installation and image/video-only capabilities.](models.md)

Commands outside Docker assume an activated Python 3.11+ virtual environment: create it with `python3 -m venv .venv`, then run `source .venv/bin/activate`. Use `python` for installation and execution in that environment; Docker commands use `python3`. Shell scripts also accept `PYTHON=/path/to/venv/bin/python`. Historical execution records retain their original commands.

# Transformers backend (no vLLM required)

> Historical results use their recorded configurations. See [current validation](validation_current.md) for later checks and settings that remain unverified.

This backend uses `Qwen3OmniMoeThinkerForConditionalGeneration` directly. It never
loads Talker, invokes `generate()`, adds a learned head, updates weights, or
requires vLLM. PyTorch SDPA uses its installed kernels; no custom compilation
step is required. Use official local Qwen3-Omni weights unchanged.

## Numerical profile and batching

The default is `--numerics stable`: official BF16 weights remain unchanged;
text activations, projection arithmetic and routing use FP32, and media items
are encoded independently of question batch size. This costs more compute than
`--numerics native`. Native BF16 remains available to reproduce older results,
but is not batch/cache invariant. Compare each profile to its own serial baseline.
No vLLM import, custom CUDA kernel or compilation is used by either HF profile.
Details and measured scope: [stability](stability.md).

```bash
python demo_hf.py --model /path/to/model --input task.json \
  --numerics stable --question-batch-size 3 --prefix-cache --projection full
```

## Install

Linux, Python 3.11+, and system FFmpeg (for audio tracks in videos); install an appropriate PyTorch binary for your hardware first.
From this checkout:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf]'
```

Python 3.11 or newer is required because the pinned PyAV 18.1.0 wheel does not support Python 3.10.

Pinned core versions are Transformers 5.13.1, PyTorch 2.11.0 and Torchvision 0.26.0. The official video processor requires Torchvision; the HF extra installs it explicitly.

The HF extra also pins TorchCodec 0.11.0+cpu, compatible with PyTorch 2.11, for the official Qwen video reader. This supports the public mini AV1 clip that the bundled Decord decoder cannot read. The public HF evaluation explicitly sets `FORCE_QWENVL_VIDEO_READER=torchcodec`; system FFmpeg shared libraries are required. Install the CPU codec wheel first from the official PyTorch index as shown above; it avoids extra CUDA video-library dependencies. Model inference still uses CUDA. Do not use the CPU index for the subsequent PyTorch/HF installation. See the [upstream compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions).

`vllm` is now an
optional extra (`pip install -e '.[vllm]'`) for the legacy/integrated vLLM paths.
Do not set MJEV_ENABLE or MJEV_ENABLE_PATCHES for HF. An existing Python
installation containing vLLM is not required or consulted by this backend.

## Demo

```bash
python demo_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --input examples/single.json --check-only
python demo_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --input examples/single.json --mode causal --output outputs/hf-causal.json
python demo_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --input examples/single.json --mode isolated --output outputs/hf-isolated.json
```

For audio/video use an input JSON such as:

```json
{
  "modality": "audio_video",
  "media_path": "clip.mp4",
  "video_options": {"fps": 1, "min_pixels": 3136, "max_pixels": 50176, "max_frames": 32},
  "questions": [
    {"id": "q1", "question": "Is anyone speaking?", "candidates": ["yes", "no"]},
    {"id": "q2", "question": "What is moving?", "candidates": ["a person", "a vehicle", "neither"]}
  ]
}
```

`modality` is image, audio, video, or audio_video. Paths are relative to the JSON.
The example options sample frames from the full video; they do not crop using
answer evidence. Set them explicitly for reproducible comparisons. `--check-only`
loads only the official processor/tokenizer and enforces the actual full input
length (default 4000). Overlength inputs raise; nothing is silently truncated.
Do not assume old benchmark token counts still apply if processing settings change.

Output follows the main package schema: candidate label/text/token_id/raw_logit/
probability, deterministic decision, tie metadata and actual prompt token count.
Softmax is only over supplied candidates; these probabilities are not calibrated.
Labels are validated as single tokens after the actual native `Answer:` suffix.
Duplicate candidates are rejected consistently with the main reference protocol.

## Implementation and boundaries

Official processing and MRoPE run on the original 2D attention mask. A temporary
hook injects a 4D additive visibility mask at the Thinker text-model boundary,
after MRoPE is computed. The mask follows the selected mode: `causal` retains
ordinary causal visibility; `isolated` lets candidates see the common prefix and
their own causal history. Decision suffix tokens see all candidates. Only eager/SDPA text
attention is supported, not FlashAttention for this custom mask path.

At the unchanged LM head, only the final hidden position is read. Default
`--projection selected` multiplies selected original weight rows in FP32;
`--projection full` computes the original head at that final position and then
selects labels. BF16 projection rounding can differ; use this switch as a
parity diagnostic, not a claim of bit-identical behavior on every device.
Hooks are removed even on failure and calls on one engine are serialized.
Use device_map=auto for multi-GPU placement; this is not vLLM tensor parallelism.
The HF wrapper adds the official text decoder layer class to `_no_split_modules`:
Transformers 5.13.1 otherwise splits some residual blocks across devices. This
changes placement only, not model forward math or weights. Python callers can
pass `max_memory={0: "20GiB", 1: "20GiB", 2: "20GiB", 3: "20GiB"}` to reserve
activation/cache headroom; adjust for the actual hardware. Stable automatic
placement reserves 4 GiB per visible GPU when no explicit memory budget is supplied.
No quantization or disk-offload support has been validated.

By default multi-question scoring reuses decoded media but recomputes each full
input. Add `--prefix-cache` to the demo or evaluator to prefill media and public
context once, then forward only each question's suffix using an independent KV
copy. Both causal and isolated modes support this. Use `--question-batch-size 3` for a true three-row model forward. Python callers
use `batch_size=3` on `score_many` or `score_questions`. Calls/groups remain
serialized on an engine; questions inside a group are batched. This is explicit
per-media reuse, not an automatic global cache.
No vLLM scheduler or Tree-KV optimization is claimed for the HF backend.
No GPU services are stopped by any HF script.

## Experimental explicit prefix cache

```python
from mjev.hf import HFMJevEngine
engine = HFMJevEngine("/path/to/Qwen3-Omni-30B-A3B-Instruct")
context = engine.prepare_context("clip.mp4", modality="audio_video",
    video_options={"fps": 1, "min_pixels": 3136, "max_pixels": 50176, "max_frames": 32})
results = engine.score_questions(context, [
    {"id": "q1", "question": "Is anyone speaking?", "candidates": ["yes", "no"]},
    {"id": "q2", "question": "Is music audible?", "candidates": ["yes", "no"]},
], mode="isolated", batch_size=3)
del context  # release retained media features and prefix KV
```

For the JSON demo: `python demo_hf.py --model /path/to/model --input task.json
--mode isolated --prefix-cache` (one shell line). The evaluator accepts the same
`--prefix-cache` flag. `score_many(..., use_prefix_cache=True)` is the convenience API.
`num_cached_tokens` reports the actual reused public-prefix length per question.

The boundary is computed from official tokenizer offsets before question text;
BPE tokens that cross that boundary stay in the question suffix. Native MRoPE
positions are calculated from each complete processed input and checked against
the cached prefix. Media features, temporal metadata and prefix tokens are
checked on reuse. Candidate masks include cached keys with the correct offsets.
The unchanged media encoders run during prefill, not during question forwards.
Processor work is still repeated per question using the retained decoded media.

Contexts are engine-local, held in memory and must be treated as opaque; do not
modify their fields or model weights/device/dtype while retaining one. Create a
new context when media, sampling settings, prompt or model changes. There is no
disk persistence, eviction policy, or zero-copy Tree-KV.
Cloning KV trades extra memory/bandwidth for branch isolation: the base cache and
each batch row's growing cache coexist. Full-model speed and memory savings require
GPU measurement; short inputs or one question may not benefit. Calls are
serialized on each engine. Length limits still apply to the **full** input.

## CPU tests

```bash
pip install pytest
python -m pytest -q tests/hf
```

Tests use actual tiny random-weight Qwen3-Omni Thinker forwards, including
image/audio/video, candidate-isolation intervention with a causal positive
control, selective/full projection, output probabilities, unchanged weights,
hook cleanup, cached/uncached logits parity, media-encoder bypass, question-order
invariance, independent KV branches and stale-prefix rejection. These are implementation witnesses, not 30B model evaluation.
The initial HF implementation passed 25 HF tests and an earlier broader
repository run passed 59 tests. Those are historical counts; the suite has
since expanded. Current stability validation is recorded in [numerical profiles](stability.md). A separate CPU
check paired the official processor and real Clotho-AQA, MuChoMusic, Video-MME-v2
and MMOU media with a tiny random Thinker (including DeepStack): two questions
per media, both modes, maximum absolute cached/uncached logit difference
5.96e-8. This is a numerical implementation check, not an accuracy measurement
or evidence of full-model BF16 parity. Official checkpoint processor tests are
separate from GPU/full-weight validation.
A full-weight, uncached paired evaluation has now completed on 1,249 accepted
questions across images, audio and video. See [paired evaluation](paired_backends.md)
for the measured scope and results. Sustained concurrent throughput and exact
backend numerical parity are not established. The historical native-BF16 cache
check below failed acceptance; stable-profile regression is recorded separately
in [numerical profiles](stability.md).

## Historical native-BF16 validation: cache acceptance failed

A corrected run on 2026-09-26 loaded the native official BF16 Thinker across four
24 GB NVIDIA GPUs, using complete decoder-layer placement and selected-logit SDPA
scoring. Loading reported zero missing/mismatched Thinker keys; unused
Talker/Code2Wav weights were ignored. A checkpoint witness compared 288 complete
expert tensors (two experts, three tensors each, in all 48 layers) directly with
checkpoint files and all matched. Metadata and sampled tensor hashes were saved;
this is not a checksum of every checkpoint tensor.

The fixed real-data sample contained four media files and eight original-label
questions, scored in causal and isolated modes. Original cached/uncached answers
agreed in **14/16** comparisons: causal 8/8, isolated 6/8. The two disagreements
were one Video-MME-v2 question and one MMOU question. Question-order and permuted-
input cached/uncached **answer labels** agreed in all 24 additional comparisons.
Question-order logits were identical, but all eight permutation comparisons
failed the raw-logit tolerance (three also failed probability tolerance). This does
not mean the model is invariant to changing candidate labels. Across 40 checks,
max absolute logit difference was 1.126003 and probability difference 0.126645,
exceeding preset 0.1 / 0.02 tolerances. The run completed mechanically but
**failed acceptance**, retaining ERROR.json and no success marker.

The cause of full-weight differences is unresolved; do not attribute it solely
to BF16 rounding. This historical result applies to `numerics="native"`,
which still emits a warning for prefix caching. See [stable numerics](stability.md)
for the new computation profile and its separate validation. Use the
uncached path for reference scoring; do not claim cache equivalence, unchanged
answers, benchmark accuracy, or production readiness from this test.

Warmed local times for two questions, including preprocessing and cache creation
but excluding model load (one observation per cell, not HTTP/API latency):

| Media sample | Mode | No KV reuse | With KV reuse |
| --- | --- | ---: | ---: |
| Clotho-AQA | causal / isolated | 0.642 / 0.703 s | 0.816 / 0.803 s |
| MuChoMusic | causal / isolated | 0.638 / 0.655 s | 0.825 / 0.820 s |
| Video-MME-v2 | causal / isolated | 7.059 / 7.634 s | 6.741 / 6.779 s |
| MMOU | causal / isolated | 1.353 / 1.454 s | 1.443 / 1.396 s |

Given the answer mismatches and tiny timing sample, this is not evidence for a
correctness-preserving speedup. Peak PyTorch allocated memory per GPU was about
15.78, 16.78, 16.78 and 12.69 GiB, distinct from driver/reserved VRAM.

The earlier v2 attempt used a subclass that bypassed native MoE weight conversion,
so its newly initialized expert weights invalidate **all official-model claims**
from that run. It is retained only for diagnosis and explicitly invalidated.
The loader now keeps the exact native class and aborts on missing/mismatched
Thinker weights or unexpected non-Talker/Code2Wav keys. The regression test uses
unfused per-expert checkpoint layout and verifies every tiny-model tensor plus
rejection of an incomplete checkpoint.

Reproduce on four free GPUs with a new output directory:

```bash
python benchmarks/integrated/validate_hf_gpu.py --model /path/to/model --root /path/to/benchmark --out outputs/hf-gpu-check --numerics native --model-revision "$MODEL_REVISION"
```

This saves loading info, source hashes, a sampled checkpoint witness, the frozen
manifest, raw scores, timings and acceptance statistics. Tolerance failures stop
with preserved results rather than relaxing the gate.

## Dataset evaluator

```bash
pip install -e '.[hf,benchmark]'
python benchmarks/integrated/evaluate_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --root /path/to/benchmark --out outputs/hf-causal --mode causal --numerics stable --model-revision "$MODEL_REVISION" --projection selected --question-batch-size 1
```

Use a new output directory per run. It reuses Accuracy/NLL/Brier/ECE from main
and the same original-label MCQ eligibility policy as the integrated vLLM runner.
Clotho's 374 native/disagreeing questions remain explicitly excluded; original
questions and references are not regenerated. Errors stop the run with a saved
ERROR.json and completed predictions. Video options are explicit in the runner;
HF video resizing is disabled after qwen-omni-utils applies those sizes, avoiding
a second resize to unrelated processor defaults. This is a reproducible new HF
sampling configuration, not proof of identical legacy vLLM token lengths.

> Set MODEL_REVISION to the actual 40-character commit of your local model. The native commands are an explicit rerun profile, not a reconstruction of missing historical settings or a guarantee of the old scores. For a publicly downloadable workflow, use [public mini evaluation](reproduce.md).
