<p align="center">
  <img src="assets/mjev-cover-somark.svg" alt="mJev — Jev, with senses." width="100%">
</p>

<h1 align="center">mJev</h1>
<p align="center"><a href="README.zh-CN.md">简体中文</a> · <strong>English</strong></p>
<p align="center"><strong>Jev, with senses. mjev-doc, with documents.</strong></p>
<p align="center"><a href="#quick-start">🚀 Quick Start</a> · <a href="https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD">🤗 mJev weights</a> · <a href="#mjev-doc">📄 mjev-doc model guide</a></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modalities-Text%20%7C%20Vision-8b5cf6" alt="Text and Vision">
  <img src="https://img.shields.io/badge/Backends-HF%20%7C%20vLLM-0891b2" alt="HF and vLLM">
</p>

## Decision intelligence beyond text

Bring an image, a document or a video. Ask several questions, define the choices, and get a decision for each one — together with every candidate's raw logit and probability.

mJev brings the Jev workflow to multimodal inputs, with trained models for general vision and document understanding. Questions share the media context while keeping their own candidate sets.

- **One context, multiple decisions:** reuse prefix KV and batch question branches.
- **Choices you define:** variable candidate sets, direct LM Head scoring and structured outputs.
- **Start with HF:** run without vLLM or custom CUDA compilation; an optional vLLM backend is available.

## 🧩 Choose your model

| Model | Best for | Get started |
| --- | --- | --- |
| **[mJev-Qwen3-VL-4B-RLCD](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD)** | General image and video decisions | [Weights](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) · [Model guide](docs/models.md) |
| **mjev-doc** | Document categories, visible attributes, field relationships and decisions under explicit criteria | [Document guide](docs/docjev/README.md) · [Real example](examples/docjev/README.md) |

Both models build on Qwen3-VL-4B-Instruct and share the HF scoring core. For audio or video with audio, use the official Qwen3-Omni Thinker backend with the [same question/candidate interface](docs/hf.md).

<a id="mjev-doc"></a>

### 📄 mjev-doc

A page can support more than one useful decision. mjev-doc adapts the workflow to document understanding through RLCD training, freezing vision and updating language parameters. Its [training and evaluation](docs/docjev/results.md) stay in this project.

Weights will be released on **Hugging Face**. The demo currently accepts a complete local mjev-doc checkpoint or the official Qwen3-VL base model.

## 🎯 Results at a glance

| Model | Evaluation cohort | Base Qwen | After training | Gain |
| --- | --- | ---: | ---: | ---: |
| mJev-Qwen3-VL-4B-RLCD | [Compositional-VQA](https://huggingface.co/datasets/Immortal-Zhang/mJev-Compositional-VQA) · 195 questions | 77.95% | **80.00%** | +2.05 pp |
| mjev-doc | [docjev-bench](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench) · 1,512 questions | 82.04% | **84.36%** | +2.31 pp |

The mJev row retains the upstream study's 195-question cohort. mjev-doc questions have Chinese and English versions; accuracy is measured over 3,024 language records. These are separate evaluation sets, and the scores measure agreement with their reference answers. [mJev training](docs/training.md) · [mjev-doc protocol and complete results](docs/docjev/results.md)

### ⚡ Same context. Keep the questions coming.

For one video and 16 questions, prefix KV reuse reduced total latency from **32.62 s to 6.29 s — 5.19× faster** in the recorded controlled experiment.

[![HF prefix-cache latency comparison](assets/cache-latency.svg)](docs/cache_scaling.md)

Measured with official Qwen3-VL-4B, HF stable/causal scoring and a 2,266-token shared prefix on one 24 GB GPU. Timings include media processing and fresh prefill, excluding model loading. [All configurations, memory usage and raw records](docs/cache_scaling.md)

<a id="quick-start"></a>

## 🚀 Quick Start

mJev and mjev-doc share the HF inference core, with **one 24 GB NVIDIA GPU** as the common deployment configuration. Use Linux, Python 3.11+ and compatible NVIDIA drivers; video inputs also need system FFmpeg.

```bash
git clone https://gitlab.soulcode.cn/immortal/mjev.git
cd mjev
python3 -m venv .venv
source .venv/bin/activate
```

### General vision · mJev

```bash
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub

export MODEL_DIR="$HOME/models/mJev-Qwen3-VL-4B-RLCD"
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir "$MODEL_DIR"

CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model "$MODEL_DIR" \
  --input examples/multiple.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-demo.json
```

### Document understanding · mjev-doc

Supply your complete mjev-doc checkpoint directory:

```bash
python -m pip install -e '.[docjev]'
CUDA_VISIBLE_DEVICES=0 python demo_docjev.py --model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --device-map cuda:0 \
  --output outputs/document-demo.json
```

🎉 Results are saved in `outputs/`. The image example uses stable numerics and prefix reuse; the document example uses native BF16 scoring. For document caching, add `--numerics stable --prefix-cache --question-batch-size 2`. [More examples](examples/README.md) · [HF deployment](docs/hf.md) · [vLLM deployment](docs/vllm.md)

## 📝 Bring your own questions

One media input, a shared context and a list of questions:

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

Media paths resolve relative to the input JSON file. Each output contains `candidates`, `decision` and `probability_sum`. See the [runnable inputs and recorded outputs](examples/README.md), including a [real document with bilingual questions](examples/docjev/README.md).

Candidate probabilities use softmax over the supplied labels; the decision selects the highest raw logit. They are relative to the question's choices, not calibrated confidence scores. Labels are checked as single tokens in the actual template. [Document input/output](docs/docjev/input-output.md) · [Audio/video inputs](docs/hf.md#demo)

## 🔎 Under the hood

```text
Media + shared context → official processor and chat template → shared prefix
                                                              ├─ Question 1 + candidates → Answer logits
                                                              ├─ Question 2 + candidates → Answer logits
                                                              └─ Question N + candidates → Answer logits
Candidate-label logits → candidate softmax → decision
```

The model's existing LM Head provides the scores. HF uses direct forward calls; vLLM uses pooling. Ordinary causal attention is the default. Candidate isolation is available explicitly for controlled experiments. [Architecture and caching](docs/docjev/architecture.md) · [Implementation guide](docs/development.md)

## 📚 Go further

| Looking to… | Start here |
| --- | --- |
| Deploy or choose a model | [Models](docs/models.md) · [HF](docs/hf.md) · [vLLM](docs/vllm.md) |
| Train or evaluate document models | [mjev-doc](docs/docjev/README.md) · [Training](docs/docjev/training.md) · [Results](docs/docjev/results.md) |
| Prepare data or reproduce evaluations | [Benchmarks](docs/benchmark.md) · [Public mini](docs/reproduce.md) |
| Explore the code or run tests | [Developer guide](docs/development.md) · [Tests](docs/testing.md) |
| Check performance and evidence | [Cache scaling](docs/cache_scaling.md) · [Current validation](docs/validation_current.md) |

Browse the [full documentation](docs/index.md) for numerical profiles and experimental backends.

## 🤝 Contribute

Bug reports, examples and improvements are welcome. [Open an issue](https://gitlab.soulcode.cn/immortal/mjev/-/issues) · [Send a merge request](https://gitlab.soulcode.cn/immortal/mjev/-/merge_requests) · [Contribution guide](CONTRIBUTING.md)

Thanks to [Immortal-Zhang](https://github.com/Immortal-Zhang), [Kyousuke661](https://github.com/Kyousuke661) and [BinyangQiu](https://github.com/BinyangQiu).

## 📜 License

Code is [Apache-2.0](LICENSE), with attribution in [NOTICE](NOTICE). Model weights and third-party data retain their own licenses. The document example keeps its CC-BY-2.0 attribution; see [third-party licenses](THIRD_PARTY_LICENSES.md).
