<p align="center">
  <img src="assets/mjev-cover-somark.svg" alt="mJev — Jev, with senses." width="100%">
</p>

<h1 align="center">mJev</h1>
<p align="center"><a href="README.zh-CN.md">简体中文</a> · <strong>English</strong></p>
<p align="center"><strong>Jev, with senses. mjev-doc, with documents.</strong></p>
<p align="center"><a href="https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD">🤗 mJev weights</a> · <a href="docs/docjev/README.md">📄 mjev-doc model guide</a></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modalities-Text%20%7C%20Vision-8b5cf6" alt="Text and Vision">
  <img src="https://img.shields.io/badge/Backends-HF%20%7C%20vLLM-0891b2" alt="HF and vLLM">
</p>

## Decision intelligence beyond text

**Jev-style decision intelligence beyond text — multimodal state in, typed decisions out.**

mJev turns shared context into explicit choices. Supply the context, ask multiple questions, and define the candidates for each one. Each result includes a decision, candidate probabilities and raw logits, making the choice easy to inspect and use downstream.

- **Shared context:** reuse prefix KV across questions and batch question branches.
- **Defined outputs:** keep a candidate set per question so every decision maps to a supplied choice.
- **Runnable workflow:** combine input processing, candidate scoring and cache-consistency checks on HF/vLLM. Start with standalone HF.

<a id="quick-start"></a>

## 🚀 Quick Start

**Validated on one 24 GB NVIDIA GPU.** Requires Linux, Python 3.11+, compatible NVIDIA drivers and system FFmpeg.

```bash
git clone https://gitlab.soulcode.cn/immortal/mjev.git
cd mjev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub

export MODEL_DIR="$HOME/models/mJev-Qwen3-VL-4B-RLCD"
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir "$MODEL_DIR"

# Run the bundled image with two questions
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model "$MODEL_DIR" \
  --input examples/multiple.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-demo.json
```

🎉 Find your results in `outputs/image-demo.json`. The default is **HF + causal + stable**; no vLLM installation or custom CUDA compilation needed.

Choose your GPUs with `CUDA_VISIBLE_DEVICES`. More options: [HF deployment](docs/hf.md) · [vLLM deployment](docs/vllm.md).

## 🧩 Choose a model

Two trained 4B models, one candidate-scoring workflow: **mJev** for general visual decisions, **mjev-doc** for document understanding. The official Omni backend also supports audio/video experiments.

| Model | Inputs | Start here |
| --- | --- | --- |
| **[mJev-Qwen3-VL-4B-RLCD](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD)** | Image + text, video + text | [Weights](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) · [Installation, demo and evaluation](docs/models.md) |
| **mjev-doc** | Document image + text | [Document demo, local checkpoint and results](docs/docjev/README.md) |
| **Qwen3-Omni-30B-A3B-Instruct** | Image, audio, video, video with audio + text | [Omni deployment guide](docs/hf.md) |

The official config selects the model family automatically. Start with 4B for image/video; choose Omni for audio. Keep the same question and candidate format.

## 📄 Meet mjev-doc

*A page full of information. More than one question worth asking.*

mjev-doc brings document images into the same Jev decision workflow. Ask about document categories, visible attributes, field relationships or explicit business criteria, and inspect every candidate score. The document adapter reuses **mJev's HF processor, attention, LM Head and prefix-cache core**; its training and evaluation modules stay separate.

**mjev-doc** starts from official Qwen3-VL-4B-Instruct and trains the language model on **1,223 document images / 15,658 bilingual language records**, with vision frozen. The first 500-record validation comparison improves from **73.0% to 77.8%**. [Training and results →](docs/docjev/results.md)

```bash
python -m pip install -e '.[docjev]'
# Supply your complete exported mjev-doc checkpoint directory.
python demo_docjev.py --model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --output outputs/docjev.json
```

| mjev-doc cohort | Base Qwen | Trained mjev-doc |
| --- | ---: | ---: |
| First validation · 500 language records | 73.0% | 77.8% |
| Unified diagnostic · 3,024 language records | 82.04% | 84.36% |

The unified evaluation includes **all 1,492 multitask questions plus the 20 added visual questions**, each in Chinese and English, with raw candidate logits and temperature-1 softmax probabilities saved for both models. Scores measure reference-label agreement; paper questions are positives only. [Counts, probabilities and protocol →](docs/docjev/results.md)

mjev-doc weights are loaded locally; they are not bundled or published by this repository. You can also try this demo with the official Qwen3-VL base model. [Get started with documents →](docs/docjev/README.md)

## ⚡ Same context. Keep the questions coming.

Same video, 16 questions: in the recorded controlled test, prefix reuse cut total latency from **32.62 s to 6.29 s — a 5.19× speedup**.

[![HF cache latency comparison: ordinary batching versus prefix KV reuse](assets/cache-latency.svg)](docs/cache_scaling.md)

Caching has a sweet spot: longer shared context and more questions. One question can be slower. For the long-prefix, 16-question case, peak allocated memory rose from **15.80 to 20.60 GiB**. This is a controlled result on one project-created video, not a universal speed guarantee.

### 📏 Performance test details

| Shared prefix | Questions | Ordinary batch | Prefix KV reuse | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 63 tokens | 3 | 0.358 s | 0.341 s | 1.05× |
| 2,266 tokens | 1 | 2.467 s | 2.631 s | 0.94× |
| 2,266 tokens | 8 | 16.464 s | 4.246 s | **3.88×** |
| 2,266 tokens | 16 | 32.620 s | 6.287 s | **5.19×** |

One 24 GB NVIDIA GPU, Qwen3-VL-4B, HF `stable` (BF16 weights, FP32 text computation), `causal`, full projection; five-run means after two warmups, with the allocator cleared before each call. Timings include media processing and fresh prefix prefill, excluding model loading. Speedup is ordinary-batch time divided by cached time; below 1 means slower.

[All eight configurations and raw records](docs/cache_scaling.md)

## 🎯 Put it to the test

On the upstream **195-question historical evaluation cohort** of [mJev-Compositional-VQA](https://huggingface.co/datasets/Immortal-Zhang/mJev-Compositional-VQA), GRPO fine-tuning improves accuracy from **77.95% to 80.00% (+2.05 percentage points)**, with four more questions answered correctly.

| Model | Correct / total | Accuracy |
| --- | ---: | ---: |
| Qwen3-VL-4B-Instruct (before GRPO) | 152 / 195 | 77.95% |
| **[mJev-Qwen3-VL-4B-RLCD](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) (after GRPO)** | **156 / 195** | **80.00%** |

Both models were evaluated on the same 195 questions, held out from RL training. Accuracy is the number of correct answers divided by the total number of questions. The public dataset provides images, questions, candidate choices and reference answers.

The released checkpoint uses GRPO for candidate selection. See [training and reward design](docs/training.md).

The mJev numbers above are retained from the upstream study. mjev-doc evaluations use different cohorts and are reported separately; these scores do not rank the two trained models against each other.

## 📝 Use your own data

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

For audio/video, use `modality` and `media_path`. Supported modalities are `image`, `audio`, `video` and `audio_video`. Configure video sampling explicitly; see [audio/video inputs](docs/hf.md#demo). General mJev fixtures are project-created; the [mjev-doc example](examples/docjev/README.md) includes one attributed CC-BY-2.0 validation table.

Output is a JSON array, one result per question, containing `candidates`, `decision` and `probability_sum`. Browse [runnable examples and recorded outputs](examples/README.md).

Actual results also include token IDs, tie metadata, input length and cache diagnostics. Probabilities are normalized only over the supplied candidates and **are not calibrated confidence scores**. Ties select the first candidate in input order.

Candidate counts may vary between 2 and 128, subject to single-token label validation in the actual template; not every count is guaranteed to work. Empty or duplicate candidates and invalid control tokens are rejected. HF limits the complete input to 4000 tokens by default. Overlength inputs raise an error rather than being silently truncated.

## 🔎 How it works

```text
Media + shared context → official processor / chat template → native model prefix prefill
                                                              ├─ Question 1 + candidates → Answer logits
                                                              ├─ Question 2 + candidates → Answer logits
                                                              └─ Question N + candidates → Answer logits
Candidate-label logits → softmax over supplied candidates → argmax decision
```

Questions share the media, but each has its own set of choices. By default (`causal`), the model reads the choices in order. For experiments, `isolated` gives each choice the shared context and question while keeping the other choices out of view. The final answer still considers all choices.

The model's output layer provides the scores. mJev returns the selected answer and candidate probabilities. See the [implementation guide](docs/development.md) for the HF and vLLM details.

## 📚 Documentation

| Looking to… | Start here |
| --- | --- |
| Deploy or choose a model | [HF](docs/hf.md) · [vLLM](docs/vllm.md) · [Models](docs/models.md) |
| Run document tasks, RLCD or model comparisons | [mjev-doc](docs/docjev/README.md) · [Results](docs/docjev/results.md) |
| Browse runnable examples | [Example guide](examples/README.md) |
| Explore the code or run tests | [Developer guide](docs/development.md) · [pytest](docs/testing.md) |
| Prepare data or reproduce an evaluation | [Benchmarks](docs/benchmark.md) · [Public mini](docs/reproduce.md) |
| Check evidence and boundaries | [Current validation](docs/validation_current.md) · [Installation checks](docs/installation_validation.md) · [Numerics](docs/stability.md) |
| Explore experimental features | [HTTP / Tree-KV](docs/integrated.md) · [AV Docker image](experiments/av/README.md) |

## 🤝 Contribute

Bug reports, examples and small fixes are welcome. [Open an issue](https://gitlab.soulcode.cn/immortal/mjev/-/issues) · [Send a PR](https://gitlab.soulcode.cn/immortal/mjev/-/merge_requests) · [Contribution guide](CONTRIBUTING.md). Include your environment, settings and a minimal reproduction when reporting a bug.

Thanks to [Immortal-Zhang](https://github.com/Immortal-Zhang), [Kyousuke661](https://github.com/Kyousuke661) and [BinyangQiu](https://github.com/BinyangQiu).

## 📜 License

Code: [Apache-2.0](LICENSE); upstream attribution: [NOTICE](NOTICE). Download official model weights separately under their own licenses. Third-party data keeps its original terms and **is not relicensed with the code**; see [third-party licenses](THIRD_PARTY_LICENSES.md). Model weights and full third-party datasets are not distributed here. The mjev-doc validation table retains its CC-BY-2.0 attribution.
