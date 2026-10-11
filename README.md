<p align="center">
  <img src="assets/mjev-hero-story.svg" alt="SoMark | mJev — Decision intelligence beyond text. One visual input, multiple inspectable decisions." width="100%">
</p>

<h1 align="center">SoMark｜mJev</h1>
<p align="center"><b>Multimodal state in. Typed decisions out.</b></p>

<p align="center"><a href="README.zh-CN.md">简体中文</a> &nbsp;&nbsp; <b>English</b></p>
<p align="center">
  <a href="https://huggingface.co/SoMarkAI/mjev-doc">🤗 Get mJev-Doc</a> &nbsp;&nbsp;
  <a href="#start">🚀 Quick start</a> &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Immortal-Zhang/docjev-bench">DocJev-bench</a>
</p>

---

## 👀 Decision intelligence beyond text

**Understand the content. Make a clear choice.** mJev brings Jev-style decisions beyond text: share an image, document or video across multiple questions, each with its own candidates. The model returns a choice, relative candidate probabilities and raw logits, connecting visual understanding to practical workflows.

**mJev-Doc focuses on documents.** What is this page? Do its fields agree? Are the stated conditions met? What happens next? Keep asking about the same document.

- **⚡ Keep the context**: reuse prefix KV across questions and batch them together.
- **🧩 Bring your choices**: every question gets its own little crew of candidates.
- **🔎 Inspect the decision**: the selected answer, every candidate probability, and raw logits.

---

> If this project helps you, please click the ⭐ Star in the top right corner — it's the greatest encouragement for the developers!

<a id="models"></a>
<a id="mjev-doc"></a>

## 🧩 01 / For documents, start with mJev-Doc

<table>
<tr>
<td width="50%" valign="top">
  <h3>📄 mJev-Doc</h3>
  <p><b>Take document understanding into business decisions.</b></p>
  <p>Classification, field relationships, completeness, and review or routing under <strong>explicitly supplied rules</strong>.</p>
  <a href="https://huggingface.co/SoMarkAI/mjev-doc"><b>Get the model ↗</b></a>
</td>
<td width="50%" valign="top">
  <h3>🖼️ mJev</h3>
  <p><b>Turn images and video into choices.</b></p>
  <p>Scenes, objects, actions, order. Ask multiple questions about shared visual context, each with its own candidates.</p>
  <a href="https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD"><b>Get the model ↗</b></a>
</td>
</tr>
</table>

Both models use **Qwen3-VL-4B-Instruct**, with Hugging Face and vLLM inference. Audio and video with audio use the separate [Qwen3-Omni backend](docs/hf.md).

## 🎯 02 / Results you can check

<table>
<tr>
<td width="33%" align="center">
  <h2>+2.31 pp</h2>
  <p>mJev-Doc · 3,024 language records<br>82.04% → 84.36%</p>
</td>
<td width="33%" align="center">
  <h2>+2.05 pp</h2>
  <p>mJev · 195 questions<br>77.95% → 80.00%</p>
</td>
<td width="34%" align="center">
  <h2>5.19×</h2>
  <p>16 questions with a shared long prefix<br>32.62 s → 6.29 s</p>
</td>
</tr>
</table>

From the [mJev-Doc diagnostic evaluation](docs/docjev/results.md), [mJev evaluation](docs/training.md) and [single-GPU 24 GB cache test](docs/cache_scaling.md), respectively.

<p>Model scores use their separate evaluation cohorts and supplied references. Chinese/English document records are paired; the diagnostic is not an independent human-annotated test set. The speedup applies to the stated official Qwen3-VL-4B test configuration.</p>

**[DocJev-bench](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench)**: 220 document images · 1,512 bilingual question pairs · 21 task types. [Explore the tasks and business scenarios →](docs/docjev/benchmark.md)

<a id="start"></a>
<a id="quick-start"></a>

## 🚀 03 / Your document is up next

Linux · Python 3.11+ · NVIDIA GPU. Standalone Hugging Face; no vLLM installation needed.

```bash
git clone --branch docs/streamline-project-20261001 https://github.com/SoMarkAI/mJev.git
cd mJev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[docjev]' huggingface_hub
hf auth login
hf download SoMarkAI/mjev-doc --local-dir ./models/mjev-doc
CUDA_VISIBLE_DEVICES=0 python demo_docjev.py --model ./models/mjev-doc \
  --input examples/docjev/multiple.json --device-map cuda:0 \
  --output outputs/document-decisions.json
```

🎉 Your first decisions will be in `outputs/document-decisions.json`. Next, bring your own document and questions.

[Document inputs & outputs](docs/docjev/input-output.md) &nbsp;&nbsp; [mJev-Doc guide](docs/docjev/README.md) &nbsp;&nbsp; [Multi-image serving](docs/docjev/serving.md)

<details>
<summary><b>Want general images and video too? Run mJev</b></summary>

Video inputs need FFmpeg.

```bash
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir ./models/mJev
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model ./models/mJev \
  --input examples/multiple.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-decisions.json
```

[Input/output examples](examples/README.md) &nbsp;&nbsp; [Hugging Face guide](docs/hf.md) &nbsp;&nbsp; [vLLM serving](docs/vllm.md)

</details>

---

Found a bug, a new use case, or a good example to share? [Come say hello](https://github.com/SoMarkAI/mJev/issues). We would love to see your name among the contributors ✨

<p><b>Build decisions, not just descriptions.</b></p>
<p>
  <a href="docs/index.md">Documentation</a> &nbsp;&nbsp;
  <a href="docs/docjev/architecture.md">How it works</a> &nbsp;&nbsp;
  <a href="docs/docjev/training.md">Training</a> &nbsp;&nbsp;
  <a href="CONTRIBUTING.md">Contribute</a>
</p>
<p>Apache-2.0 (code) &nbsp;&nbsp; <a href="NOTICE">NOTICE</a> &nbsp;&nbsp; <a href="THIRD_PARTY_LICENSES.md">Third-party &amp; model licenses</a><br>Thanks to <a href="https://github.com/Immortal-Zhang">Immortal-Zhang</a>, <a href="https://github.com/Kyousuke661">Kyousuke661</a> and <a href="https://github.com/BinyangQiu">BinyangQiu</a>.</p>
