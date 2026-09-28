<p align="center">
  <img src="assets/mjev-banner.svg" alt="mJev — Jev, with senses." width="100%">
</p>

<h1 align="center">mJev</h1>
<p align="center"><a href="README.zh-CN.md">简体中文</a> · <strong>English</strong></p>
<p align="center"><strong>Jev, with senses.</strong></p>
<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#documentation">Docs</a> ·
  <a href="docs/validation_current.md">Validation Status</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modalities-Image%20%7C%20Video%20%7C%20Audio-8b5cf6" alt="Image, Video and Audio">
  <img src="https://img.shields.io/badge/Backends-HF%20%7C%20vLLM-0891b2" alt="HF and vLLM">
</p>

## Turn shared context into decisions

A video can raise more than one question: what moved, which object mattered, and what happened first? mJev brings those questions into one workflow. Supply the media, define the choices, and get a decision with candidate probabilities for every question.

Built for **video attribute annotation, fixed-choice visual question answering and multimodal evaluation**, mJev connects media processing, question branches, candidate scoring and cache-consistency checks on top of HF and vLLM.

- **One context, many questions.** Reuse the media prefix across questions and batch their execution.
- **Your choices, structured results.** Give each question its own candidates and receive results ready for downstream use.
- **Start small, extend across modalities.** Use Qwen3-VL-4B for images and video, or Qwen3-Omni for audio and video with sound.

## See it in action

[![Motion preview: a red ball moves right, then a blue square rises](examples/motion-demo/preview.gif)](examples/motion-demo/motion.mp4)

**[Open the 6-second video](examples/motion-demo/motion.mp4)** · [Three-question input](examples/motion-demo/input.json) · [Full recorded output](examples/motion-demo/recorded-output.json)

A red ball moves right. A blue square rises next. One video, three questions: what happened, which object moved, and what came first? These are **actual Qwen3-VL-4B outputs**.

| Question | Candidate probabilities | Decision |
| --- | --- | --- |
| How does the red ball move? | From left to right: 71.34%<br>From right to left: 28.37%<br>Upward: 0.27%<br>It stays still: 0.02% | **From left to right** |
| Which object moves upward? | The red ball: 2.33%<br>The blue square: 97.29%<br>Both objects: 0.38% | **The blue square** |
| Which movement happens first? | The red ball moves right: 89.62%<br>The blue square moves up: 10.04%<br>Both movements begin at the same time: 0.34% | **The red ball moves right** |

**Relative probabilities within the candidate set, not calibrated confidence.** Values are rounded.

Project-owned animation, run with HF on one GPU; settings and raw results are in the linked record.

<a id="quick-start"></a>

## Quick Start

**Validated on one 24 GB NVIDIA GPU. Run the example below.** Use Linux, Python 3.11+, compatible NVIDIA drivers and system FFmpeg.

```bash
git clone https://github.com/SoMarkAI/mJev.git
cd mJev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub

export MODEL_DIR="$HOME/models/Qwen3-VL-4B-Instruct"
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --local-dir "$MODEL_DIR"

# Run the video and three questions shown above
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model "$MODEL_DIR" \
  --input examples/motion-demo/input.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 3 --output outputs/motion-demo.json
```

The default entry point is **HF** (`demo_hf.py`), with `causal` attention and `stable` numerics. No Docker, vLLM installation or custom CUDA compilation is required for this example. The result is saved to `outputs/motion-demo.json`.

Use `CUDA_VISIBLE_DEVICES` to choose the GPU or GPUs available to HF; capacity depends on the model, input length and question batch size. See [deployment options](docs/hf.md) for more settings and [vLLM deployment](docs/vllm.md) for the optional backend.

<a id="performance"></a>

## When prefix reuse pays off

In our controlled 16-question video test, prefix reuse reduced total latency from **32.62 s to 6.29 s**, including prefix construction. The benefit depends on how much context the questions share:

| Shared prefix | Questions | Ordinary batch | Prefix KV reuse | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 63 tokens | 3 | 0.358 s | 0.341 s | 1.05× |
| 2,266 tokens | 1 | 2.467 s | 2.631 s | 0.94× |
| 2,266 tokens | 8 | 16.464 s | 4.246 s | **3.88×** |
| 2,266 tokens | 16 | 32.620 s | 6.287 s | **5.19×** |

<details>
<summary>Measurement conditions and memory trade-off</summary>

One 24 GB NVIDIA GPU, Qwen3-VL-4B, HF `stable` (BF16 weights, FP32 text computation), `causal`, full projection; five-run means after two warmups, with the allocator cleared before each call. Timings include media processing and fresh prefix prefill, excluding model loading. Speedup is ordinary-batch time divided by cached time; below 1 means slower.

For 16 questions with the long prefix, peak allocated GPU memory increased from **15.80 to 20.60 GiB**, including model weights. These are controlled results on one project-generated video, not a general speed guarantee. [All eight configurations, raw results and reproduction](docs/cache_scaling.md)

</details>

<a id="evaluation"></a>

## Evaluation

**[mJev-Compositional-VQA](https://huggingface.co/datasets/Immortal-Zhang/mJev-Compositional-VQA)** provides image questions with candidate choices and reference answers for compositional visual reasoning. The dataset is hosted separately on Hugging Face; consult its dataset card for access, format and per-image licensing. Results for this benchmark will be linked here when published.

## Models and backends

| Model | Inputs | Start here |
| --- | --- | --- |
| **Qwen3-VL-4B-Instruct** | Image + text, video + text | [4B installation, demo and evaluation](docs/models.md) |
| **Qwen3-Omni-30B-A3B-Instruct** | Image, audio, video, video with audio + text | [Omni deployment guide](docs/hf.md) |

The official config selects the model family automatically. Start with 4B for image/video; choose Omni for audio. Keep the same question and candidate format.

HF calls the model's `forward` directly. The optional vLLM pooling backend uses `AsyncLLM.encode` with pinned versions and custom hooks. Neither main path calls `generate()`. HTTP and Tree-KV are [experimental paths](docs/integrated.md) with separate validation scope.

<details>
<summary><strong>Input and output format</strong></summary>

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

For audio/video, use `modality` and `media_path`. Supported modalities are `image`, `audio`, `video` and `audio_video`. Configure video sampling explicitly; see [audio/video inputs](docs/hf.md#demo). Repository examples use project-created media; third-party evaluation media are hosted separately.

Output is a JSON array, one result per question, containing `candidates`, `decision` and `probability_sum`. See the `questions` field in the [actual output record](examples/motion-demo/recorded-output.json) above.

Actual results also include token IDs, tie metadata, input length and cache diagnostics. Probabilities are normalized only over the supplied candidates and **are not calibrated confidence scores**. Ties select the first candidate in input order.

Candidate counts may vary between 2 and 128, subject to single-token label validation in the actual template; not every count is guaranteed to work. Empty or duplicate candidates and invalid control tokens are rejected. HF limits the complete input to 4000 tokens by default. Overlength inputs raise an error rather than being silently truncated.

</details>

## How it works

```text
Media + shared context → official processor / chat template → native model prefix prefill
                                                              ├─ Question 1 + candidates → Answer logits
                                                              ├─ Question 2 + candidates → Answer logits
                                                              └─ Question N + candidates → Answer logits
Candidate-label logits → softmax over supplied candidates → argmax decision
```

Each question has its own candidate set. `causal` uses ordinary causal visibility. In `isolated` mode, each candidate can attend to the shared context, question and its own preceding tokens; the final `Answer:` position can attend to all candidates. Use `causal` by default, or select `isolated` for controlled attention experiments.

Question branches are independent. Candidate isolation is a separate, optional setting **within a question**; it is not required for prefix reuse. HF copies the prefix KV into question branches. The `stable` numerical profile adds compute and memory overhead; see [numerical profiles](docs/stability.md) for details.

<a id="documentation"></a>

## Documentation

| I want to… | Start here |
| --- | --- |
| Deploy a model or use the Python API | [Model selection](docs/models.md) · [HF](docs/hf.md) · [vLLM](docs/vllm.md) |
| Understand scoring, caching and numerical settings | [Developer guide](docs/development.md) · [Stability](docs/stability.md) |
| Prepare data and reproduce an evaluation | [Benchmarks](docs/benchmark.md) · [Public mini workflow](docs/reproduce.md) |
| Check what has been verified | [Current validation](docs/validation_current.md) · [Installation checks](docs/installation_validation.md) |
| Contribute code or run tests | [Contributing](CONTRIBUTING.md) · [pytest suites](docs/testing.md) |
| Explore optional runtime implementations | [HTTP / Tree-KV](docs/integrated.md) · [AV Docker image](experiments/av/README.md) |

## Community and contributions

Help shape mJev through reproducible bug reports, new use cases, documentation and controlled comparisons. Start with the [contribution guide](CONTRIBUTING.md), open an [issue](https://github.com/SoMarkAI/mJev/issues), or submit a [pull request](https://github.com/SoMarkAI/mJev/pulls).

For bug reports, include the commit, environment, backend, numerical profile and a minimal reproduction. Keep credentials, private media and model weights out of submissions.

Thanks to our contributors: [Immortal-Zhang](https://github.com/Immortal-Zhang), [Kyousuke661](https://github.com/Kyousuke661) and [BinyangQiu](https://github.com/BinyangQiu).

## License

mJev code is licensed under [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution. Download official [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) or [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) weights separately under their respective licenses.

Third-party annotations, audio and video retain their original rights and restrictions; **this project does not relicense them under Apache-2.0**. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md). Model weights and benchmark media are not distributed in this repository.
