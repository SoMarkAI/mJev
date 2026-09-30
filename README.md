<p align="center">
  <img src="assets/mjev-cover-somark.svg" alt="mJev — Jev, with senses." width="100%">
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

## 🎬 A six-second show

[![Motion preview: a red ball moves right, then a blue square rises](examples/motion-demo/preview.gif)](examples/motion-demo/motion.mp4)

**[Open the 6-second video](examples/motion-demo/motion.mp4)** · [Three-question input](examples/motion-demo/input.json) · [Full recorded output](examples/motion-demo/recorded-output.json)

Red ball goes first. Blue square takes the stage. Six seconds, three questions — did the model keep up? 👀

These are **actual Qwen3-VL-4B + HF outputs** on a project-created animation.

| Question | Candidate probabilities | Decision |
| --- | --- | --- |
| How does the red ball move? | From left to right: 71.34%<br>From right to left: 28.37%<br>Upward: 0.27%<br>It stays still: 0.02% | **From left to right** |
| Which object moves upward? | The red ball: 2.33%<br>The blue square: 97.29%<br>Both objects: 0.38% | **The blue square** |
| Which movement happens first? | The red ball moves right: 89.62%<br>The blue square moves up: 10.04%<br>Both movements begin at the same time: 0.34% | **The red ball moves right** |

**Relative probabilities within the candidate set, not calibrated confidence.** Values are rounded.

<a id="quick-start"></a>

## 🚀 Your turn

**Validated on one 24 GB NVIDIA GPU.** Requires Linux, Python 3.11+, compatible NVIDIA drivers and system FFmpeg.

```bash
git clone https://github.com/SoMarkAI/mJev.git
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

🎉 Find your results in `outputs/motion-demo.json`. The default is **HF + causal + stable**; no vLLM installation or custom CUDA compilation needed.

Choose your GPUs with `CUDA_VISIBLE_DEVICES`. More options: [HF deployment](docs/hf.md) · [vLLM deployment](docs/vllm.md).

## 🧩 Pick your teammate

| Model | Inputs | Start here |
| --- | --- | --- |
| **[mJev (Qwen3-VL-4B)](https://huggingface.co/SoMarkAI/mJev)** | Image + text, video + text | [Weights](https://huggingface.co/SoMarkAI/mJev) · [Installation, demo and evaluation](docs/models.md) |
| **Qwen3-Omni-30B-A3B-Instruct** | Image, audio, video, video with audio + text | [Omni deployment guide](docs/hf.md) |

The official config selects the model family automatically. Start with 4B for image/video; choose Omni for audio. Keep the same question and candidate format.

<details>
<summary>🔎 Curious about the moving parts?</summary>

```text
Media + shared context → official processor / chat template → native model prefix prefill
                                                              ├─ Question 1 + candidates → Answer logits
                                                              ├─ Question 2 + candidates → Answer logits
                                                              └─ Question N + candidates → Answer logits
Candidate-label logits → softmax over supplied candidates → argmax decision
```

Every question brings its own little crew of choices 🧩

- **Default: `causal`.** Choices arrive in order; later choices can read what came before.
- **Running a comparison? Try `isolated`.** Each choice gets its own booth: shared context and question, no peeking at the neighbors. The final decision still considers information from all the choices.

The model’s existing output layer does the scoring; you get a decision and candidate probabilities. Curious about the HF and vLLM plumbing? [Here are the technical details](docs/development.md).

</details>

## Training and reward

The released mJev model is fine-tuned with [GRPO](https://arxiv.org/abs/2402.03300) for candidate selection. Before training, a frozen scorer assigns each example a target probability $p^{\ast}$: the probability of the correct option within its candidate set. Each sampled answer receives:

```math
r(\hat{y}) = \begin{cases}
+p^{\ast}, & \hat{y} = y^{\ast} \\
-p^{\ast}, & \text{otherwise}
\end{cases}
```

Here $y^{\ast}$ is the correct label; incorrect or invalid answers receive the negative reward. Rewards lie in $[-1, 1]$, with higher target probabilities producing stronger signals. Outputs are constrained to one candidate label, so no separate format reward is used. Reward scaling is disabled to preserve this weighting, and a separate KL penalty limits drift from the reference model. Training updates the language model while freezing the vision tower and aligner.

## ⚡ Same context. Keep the questions coming.

Same video, 16 questions: in the recorded controlled test, prefix reuse cut total latency from **32.62 s to 6.29 s — a 5.19× speedup**.

[![HF cache latency comparison: ordinary batching versus prefix KV reuse](assets/cache-latency.svg)](docs/cache_scaling.md)

Caching has a sweet spot: longer shared context and more questions. One question can be slower. For the long-prefix, 16-question case, peak allocated memory rose from **15.80 to 20.60 GiB**. This is a controlled result on one project-created video, not a universal speed guarantee.

<details>
<summary>📏 The numbers and measurement conditions</summary>

| Shared prefix | Questions | Ordinary batch | Prefix KV reuse | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 63 tokens | 3 | 0.358 s | 0.341 s | 1.05× |
| 2,266 tokens | 1 | 2.467 s | 2.631 s | 0.94× |
| 2,266 tokens | 8 | 16.464 s | 4.246 s | **3.88×** |
| 2,266 tokens | 16 | 32.620 s | 6.287 s | **5.19×** |

One 24 GB NVIDIA GPU, Qwen3-VL-4B, HF `stable` (BF16 weights, FP32 text computation), `causal`, full projection; five-run means after two warmups, with the allocator cleared before each call. Timings include media processing and fresh prefix prefill, excluding model loading. Speedup is ordinary-batch time divided by cached time; below 1 means slower.

[All eight configurations and raw records](docs/cache_scaling.md)

</details>

## 🎯 Put it to the test

On a **195-question evaluation subset** of [mJev-Compositional-VQA](https://huggingface.co/datasets/Immortal-Zhang/mJev-Compositional-VQA), GRPO fine-tuning improves accuracy from **77.95% to 80.00% (+2.05 percentage points)**, with four more questions answered correctly.

| Model | Correct / total | Accuracy |
| --- | ---: | ---: |
| Qwen3-VL-4B-Instruct (before GRPO) | 152 / 195 | 77.95% |
| **[mJev](https://huggingface.co/SoMarkAI/mJev) (after GRPO)** | **156 / 195** | **80.00%** |

Both models were evaluated on the same 195 questions, held out from RL training. Accuracy is the number of correct answers divided by the total number of questions. The public dataset provides images, questions, candidate choices and reference answers.

<details>
<summary>📝 Bring your own input</summary>

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

</details>

## 📚 Take the shortcut

| Looking to… | Start here |
| --- | --- |
| Deploy or choose a model | [HF](docs/hf.md) · [vLLM](docs/vllm.md) · [Models](docs/models.md) |
| Explore the code or run tests | [Developer guide](docs/development.md) · [pytest](docs/testing.md) |
| Prepare data or reproduce an evaluation | [Benchmarks](docs/benchmark.md) · [Public mini](docs/reproduce.md) |
| Check evidence and boundaries | [Current validation](docs/validation_current.md) · [Installation checks](docs/installation_validation.md) · [Numerics](docs/stability.md) |
| Explore experimental features | [HTTP / Tree-KV](docs/integrated.md) · [AV Docker image](experiments/av/README.md) |

## 🤝 Pull up a chair

Found a bug? Have a use case? Even a two-line docs fix is welcome. [Open an issue](https://github.com/SoMarkAI/mJev/issues) · [Send a PR](https://github.com/SoMarkAI/mJev/pulls) · [Contribution guide](CONTRIBUTING.md). For bug reports, bring your environment, settings and a minimal reproduction.

Thanks to [Immortal-Zhang](https://github.com/Immortal-Zhang), [Kyousuke661](https://github.com/Kyousuke661) and [BinyangQiu](https://github.com/BinyangQiu). There's room for your name here, too ✨

## 📜 License

Code: [Apache-2.0](LICENSE); upstream attribution: [NOTICE](NOTICE). Download official model weights separately under their own licenses. Third-party data keeps its original terms and **is not relicensed with the code**; see [third-party licenses](THIRD_PARTY_LICENSES.md). Model weights and third-party evaluation media are not distributed in this repository.
