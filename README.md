<p align="center">
  <img src="assets/mjev-banner.svg" alt="mJev — Jev, with senses." width="100%">
</p>

<h1 align="center">mJev</h1>
<p align="center"><a href="README.zh-CN.md">简体中文</a> · <strong>English</strong></p>
<p align="center"><strong>Jev, with senses.</strong></p>
<p align="center">
  <a href="https://huggingface.co/SoMarkAI/mJev">🤗 Hugging Face Model</a> ·
  <a href="docs/index.md">Project Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="docs/hf.md">Deployment</a> ·
  <a href="docs/development.md">Developer Guide</a> ·
  <a href="docs/validation_current.md">Validation Status</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modalities-Text%20%7C%20Vision-8b5cf6" alt="Text and Vision">
  <img src="https://img.shields.io/badge/Backends-HF%20%7C%20vLLM-0891b2" alt="HF and vLLM">
</p>

## Decision intelligence beyond text

**Jev-style decision intelligence beyond text — multimodal state in, typed decisions out.**

Text and vision today. Audio, video, and more to come.

mJev turns shared context into explicit choices. Supply the context, ask multiple questions, and define the candidates for each one. Each result includes a decision, candidate probabilities and raw logits, making the choice easy to inspect and use downstream.

- **Shared context:** reuse prefix KV across questions and batch question branches.
- **Defined outputs:** keep a candidate set per question so every decision maps to a supplied choice.
- **Runnable workflow:** combine input processing, candidate scoring and cache-consistency checks on HF/vLLM. Start with standalone HF.

## Reinforcement learning for better decisions

mJev uses [GRPO](https://arxiv.org/abs/2402.03300) with a target-probability-weighted correctness reward. For each training example, a frozen pre-RL candidate scorer records $p^* = p(y^* \mid x, q, C)$: the probability assigned to the ground-truth option given the media, question and candidate set. A label-only rollout receives

$$
r(\hat{y}) =
\begin{cases}
+p^*, & \hat{y} = y^* \\
-p^*, & \hat{y} \ne y^*.
\end{cases}
$$

The reward is verifiable and bounded in $[-1, 1]$: correct labels receive a positive signal, while incorrect or invalid labels receive an equally sized negative signal. Examples with a higher frozen target probability therefore carry more weight without allowing any one example to produce an unbounded reward. Rollouts are constrained to one candidate label, so no separate format reward is needed. Reward scaling is disabled to preserve the magnitude of $p^*$, while a small reference-model KL penalty is applied separately to limit policy drift. The vision tower and aligner remain frozen; GRPO updates the language model.

## From motion to sequence: see an actual run

[![Motion preview: a red ball moves right, then a blue square rises](examples/motion-demo/preview.gif)](examples/motion-demo/motion.mp4)

**[Open the 6-second video](examples/motion-demo/motion.mp4)** · [Three-question input](examples/motion-demo/input.json) · [Full recorded output](examples/motion-demo/recorded-output.json)

A red ball moves right. A blue square rises next. One video, three questions: what happened, which object moved, and what came first? These are **actual Qwen3-VL-4B outputs**.

| Question | Candidate probabilities | Decision |
| --- | --- | --- |
| How does the red ball move? | From left to right: 71.34%<br>From right to left: 28.37%<br>Upward: 0.27%<br>It stays still: 0.02% | **From left to right** |
| Which object moves upward? | The red ball: 2.33%<br>The blue square: 97.29%<br>Both objects: 0.38% | **The blue square** |
| Which movement happens first? | The red ball moves right: 89.62%<br>The blue square moves up: 10.04%<br>Both movements begin at the same time: 0.34% | **The red ball moves right** |

**Relative probabilities within the candidate set, not calibrated confidence.** Values are rounded.

Project-owned animation, run with HF on one GPU; settings and raw results are in the linked record. The static rectangle remains available as a [minimal example](examples/video-demo/input.json) ([video](examples/video-demo/rectangle.mp4) · [results](examples/video-demo/recorded-output.json)).

<a id="quick-start"></a>

## Run on one GPU

**Validated on one 24 GB NVIDIA GPU. Run the example below.** Use Linux, Python 3.11+, compatible NVIDIA drivers and system FFmpeg.

Set `REPOSITORY_URL` to the URL from this repository’s Code/Clone button, then run:

```bash
git clone "$REPOSITORY_URL" mJev
cd mJev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub

export MODEL_DIR="$HOME/models/mJev"
hf download SoMarkAI/mJev --local-dir "$MODEL_DIR"

# Run the video and three questions shown above
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model "$MODEL_DIR" \
  --input examples/motion-demo/input.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 3 --output outputs/motion-demo.json
```

This example uses HF, without Docker, vLLM installation or custom CUDA compilation. `demo_hf.py` is the HF entry point. The root `Dockerfile` is an optional vLLM demo image; the [experimental AV image](experiments/av/README.md) is for evaluation and the experimental runtime.

**Choose your GPUs.** HF automatically places the model on visible devices: use `CUDA_VISIBLE_DEVICES=0` for one GPU or `CUDA_VISIBLE_DEVICES=0,1` for two. Capacity depends on the model, input length and concurrency. vLLM exposes `--tensor-parallel-size`, defaulting to 1 for 4B and 4 for Omni.

Need audio or the 30B model? See [model selection and installation](docs/models.md) and [Omni HF deployment](docs/hf.md). Use `python` in the activated virtual environment and `python3` inside Docker.

## Choose your model

| Model | Inputs | Start here |
| --- | --- | --- |
| **[mJev (Qwen3-VL-4B)](https://huggingface.co/SoMarkAI/mJev)** | Image + text, video + text | [Weights](https://huggingface.co/SoMarkAI/mJev) · [Installation, demo and evaluation](docs/models.md) |
| **Qwen3-Omni-30B-A3B-Instruct** | Image, audio, video, video with audio + text | [Omni deployment guide](docs/hf.md) |

The official config selects the model family automatically. Start with 4B for image/video; choose Omni for audio. Keep the same question and candidate format.

## How it works

```text
Media + shared context → official processor / chat template → native model prefix prefill
                                                              ├─ Question 1 + candidates → Answer logits
                                                              ├─ Question 2 + candidates → Answer logits
                                                              └─ Question N + candidates → Answer logits
Candidate-label logits → softmax over supplied candidates → argmax decision
```

Each question has its own candidate set. `causal` uses ordinary causal visibility. In `isolated` mode, each candidate can attend to the shared context, question and its own preceding tokens; the final `Answer:` position can attend to all candidates. Use `causal` by default, or select `isolated` for controlled attention experiments.

HF calls `forward` directly; vLLM pooling calls `AsyncLLM.encode`. Neither main path calls `generate()`.

## Evidence behind the features

| Feature | What was verified |
| --- | --- |
| Single-GPU entry point | 4B HF/vLLM image and video inference on one 24 GB NVIDIA GPU; clean HF installation and documented demo passed. |
| Cache/batch consistency | In the fixed small-scale tests, 32 repeated comparisons per backend had a maximum logit difference of zero. This measures numerical consistency. |
| Multi-question runtime | The controlled study below compares ordinary batching and prefix reuse across input lengths and question counts. |

**Reuse the context. Spend less time answering the next question.**

In our controlled 16-question video test, prefix reuse reduced total latency from **32.62 s to 6.29 s**, including prefix construction. The benefit depends on how much context the questions share:

| Shared prefix | Questions | Ordinary batch | Prefix KV reuse | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 63 tokens | 3 | 0.358 s | 0.341 s | 1.05× |
| 2,266 tokens | 1 | 2.467 s | 2.631 s | 0.94× |
| 2,266 tokens | 8 | 16.464 s | 4.246 s | **3.88×** |
| 2,266 tokens | 16 | 32.620 s | 6.287 s | **5.19×** |

One 24 GB NVIDIA GPU, Qwen3-VL-4B, HF `stable` (BF16 weights, FP32 text computation), `causal`, full projection; five-run means after two warmups, with the allocator cleared before each call. Timings include media processing and fresh prefix prefill, excluding model loading. Speedup is ordinary-batch time divided by cached time; below 1 means slower.

For 16 questions with the long prefix, peak allocated GPU memory increased from **15.80 to 20.60 GiB**, including model weights. These are controlled results on one project-generated video, not a general speed guarantee. [All eight configurations, raw results and reproduction](docs/cache_scaling.md) · [Earlier three-question fixture](docs/cache_scaling.md#earlier-three-question-fixture)

## Input and output

Media paths resolve relative to the input JSON file. See [examples/single.json](examples/single.json) for one question. Multiple questions share the media and context:

```json
{
  "image": "rectangle.png",
  "context": "Inspect the supplied picture.",
  "questions": [
    {"question": "What color is the rectangle?", "candidates": ["Red", "Blue", "Green"]},
    {"question": "Which shape is shown?", "candidates": ["Rectangle", "Circle"]}
  ]
}
```

For audio/video, use `modality` and `media_path`. Supported modalities are `image`, `audio`, `video` and `audio_video`. Configure video sampling explicitly; see [audio/video inputs](docs/hf.md#demo). Repository examples use synthetic images and include no third-party evaluation media.

Output is a JSON array, one result per question, containing `candidates`, `decision` and `probability_sum`. See the `questions` field in the [actual output record](examples/motion-demo/recorded-output.json) above.

Actual results also include token IDs, tie metadata, input length and cache diagnostics. Probabilities are normalized only over the supplied candidates and **are not calibrated confidence scores**. Ties select the first candidate in input order.

Candidate counts may vary between 2 and 128, subject to single-token label validation in the actual template; not every count is guaranteed to work. Empty or duplicate candidates and invalid control tokens are rejected. HF limits the complete input to 4000 tokens by default. Overlength inputs raise an error rather than being silently truncated.

## Deployment and developer documentation

| Goal | Documentation |
| --- | --- |
| Run the HF demo / Python API without vLLM | [HF installation, inputs and caching](docs/hf.md) |
| Docker, configurable GPU parallelism and vLLM pooling | [vLLM deployment and validation](docs/vllm.md) |
| Understand implementation responsibilities and extend scoring | [Developer guide](docs/development.md) |
| Numerical profiles and cache consistency | [Stability and validation scope](docs/stability.md) |
| Load datasets and evaluate Accuracy / NLL / Brier / ECE | [Benchmark guide](docs/benchmark.md) |
| Explore HTTP and Tree-KV implementations | [Experimental runtime](docs/integrated.md) |
| Review release checks and limitations | [Current validation](docs/validation_current.md) |

## Runtime notes

mJev is a research toolkit. The default path uses HF, `causal` attention and `stable` numerics. The stable profile adds compute and memory overhead; cache benefits depend on input length and question count. HF copies prefix KV into question branches; vLLM uses pinned versions and custom hooks.

For hardware configurations, numerical differences, experimental options and historical results, see [current validation](docs/validation_current.md) and [numerical profiles](docs/stability.md).

## Community and contributions

Contributions are welcome: reproducible bug reports, documentation fixes, mechanism tests and controlled performance comparisons. Read the [contribution guide](CONTRIBUTING.md) and [developer documentation](docs/development.md).

When reporting a problem, include the commit, environment, backend, numerical profile, minimal input and reproduction command. For code contributions, use a focused branch, run relevant checks and describe validation and limitations in your merge request. Keep credentials, private media and model weights out of submissions.

- Issues: open the repository’s Issues tab for bug reports and feature discussions
- Pull Requests: open the repository’s Pull Requests tab to contribute

## License and third-party assets

mJev code is licensed under [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution. Download official [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) or [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) weights separately under their respective licenses.

Third-party annotations, audio and video retain their original rights and restrictions; **this project does not relicense them under Apache-2.0**. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Model weights and benchmark media are not distributed in this repository.

## Reproduction and current validation

- [Public mini: prepare → infer → report](docs/reproduce.md)
- [Current validation and backend boundaries](docs/validation_current.md)
- [pytest test suites](docs/testing.md)
- [Isolated installation checks and limitations](docs/installation_validation.md)

## Contributors

Thanks to everyone contributing to mJev. Listed in no particular order:

- [Immortal-Zhang](https://github.com/Immortal-Zhang)
- [Kyousuke661](https://github.com/Kyousuke661)
- [BinyangQiu](https://github.com/BinyangQiu)
