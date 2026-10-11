<p align="center">
  <img src="assets/mjev-hero-story.svg" alt="mJev-Doc — See once. Decide many. 一份视觉输入，多个可检查的决策" width="100%">
</p>

<h1 align="center">SoMark｜mJev</h1>
<p align="center"><b>Multimodal state in. Typed decisions out.</b></p>

<p align="center"><b>简体中文</b> &nbsp;&nbsp; <a href="README.md">English</a></p>
<p align="center">
  <a href="https://huggingface.co/SoMarkAI/mjev-doc">🤗 获取 mJev-Doc</a> &nbsp;&nbsp;
  <a href="#start">🚀 快速开始</a> &nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/Immortal-Zhang/docjev-bench">DocJev-bench</a>
</p>

---

## 👀 超越文本的决策智能

**让模型看懂内容，也做出明确选择。** mJev 将 Jev 式决策带到文本之外：以图片、文档和视频作为共享上下文，提出多个问题，为每个问题定义候选项。模型返回选择、候选相对概率和原始 logits，让视觉理解接上实际工作流程。

**mJev-Doc 专注文档。** 它是什么？字段能否对上？给定条件是否满足？下一步怎么处理？围绕同一份文档，可以接着问。

- **⚡ 上下文接着用**：跨问题复用前缀 KV，支持批处理。
- **🧩 选项由你定**：每道题，带着自己的选项小队。
- **🔎 决策看得见**：选择、候选概率、原始 logits，一起交给你。

---

> 如果这个项目对你有帮助，请点击右上角的 ⭐ Star——这是对开发者最大的鼓励！

<a id="models"></a>
<a id="mjev-doc"></a>

## 🧩 01 / 文档判断，先找 mJev-Doc

<table>
<tr>
<td width="50%" valign="top">
  <h3>📄 mJev-Doc</h3>
  <p><b>让文档理解走向业务判断。</b></p>
  <p>分类、字段关系、完整性检查，以及依据<strong>明确给定规则</strong>的审核与分流。</p>
  <a href="https://huggingface.co/SoMarkAI/mjev-doc"><b>获取模型 ↗</b></a>
</td>
<td width="50%" valign="top">
  <h3>🖼️ mJev</h3>
  <p><b>让图像与视频变成选择。</b></p>
  <p>场景、目标、动作、顺序……围绕共享视觉上下文连续提问，并为每题设置不同候选。</p>
  <a href="https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD"><b>获取模型 ↗</b></a>
</td>
</tr>
</table>

两个模型均基于 **Qwen3-VL-4B-Instruct**。支持 Hugging Face 与 vLLM 推理；音频和带音轨视频可使用独立的 [Qwen3-Omni 后端](docs/hf.zh-CN.md)。

## 🎯 02 / 看表现，也看依据

<table>
<tr>
<td width="33%" align="center">
  <h2>+2.31 pp</h2>
  <p>mJev-Doc · 3,024 条语言记录<br>82.04% → 84.36%</p>
</td>
<td width="33%" align="center">
  <h2>+2.05 pp</h2>
  <p>mJev · 195 道题<br>77.95% → 80.00%</p>
</td>
<td width="34%" align="center">
  <h2>5.19×</h2>
  <p>16 问题、共享长前缀场景<br>32.62 秒 → 6.29 秒</p>
</td>
</tr>
</table>

分别来自 [mJev-Doc 诊断评测](docs/docjev/results.md)、[mJev 评测](docs/training.zh-CN.md) 与 [24 GB 单卡缓存测试](docs/cache_scaling.zh-CN.md)。

<p>两组模型分数对应各自的评测集，与其参考答案比较；mJev-Doc 的中英文记录成对对应，当前结果不代表独立的人类标注测试集。速度提升仅适用于所述官方 Qwen3-VL-4B 测试配置。</p>

**[DocJev-bench](https://huggingface.co/datasets/Immortal-Zhang/docjev-bench)**：220 张文档图片 · 1,512 对中英文问题 · 21 种任务类型。[浏览任务与业务场景 →](docs/docjev/benchmark.zh-CN.md)

<a id="start"></a>
<a id="quick-start"></a>

## 🚀 03 / 跑起来，轮到你的文档了

Linux · Python 3.11+ · NVIDIA GPU。直接使用 Hugging Face，无需安装 vLLM。

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

🎉 你的第一组决策，在 `outputs/document-decisions.json` 等你。接下来，换成自己的文档和问题吧。

[文档输入输出](docs/docjev/input-output.md) &nbsp;&nbsp; [mJev-Doc 指南](docs/docjev/README.zh-CN.md) &nbsp;&nbsp; [多图片服务](docs/docjev/serving.zh-CN.md)

<details>
<summary><b>也想测试通用图片与视频？展开运行 mJev</b></summary>

视频需 FFmpeg。

```bash
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir ./models/mJev
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model ./models/mJev \
  --input examples/multiple.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-decisions.json
```

[输入输出示例](examples/README.zh-CN.md) &nbsp;&nbsp; [Hugging Face 指南](docs/hf.zh-CN.md) &nbsp;&nbsp; [vLLM 服务](docs/vllm.zh-CN.md)

</details>

---

发现 bug、想到新场景，或者想补一个好例子？欢迎[来聊聊](https://github.com/SoMarkAI/mJev/issues)，也期待你的名字出现在贡献者里 ✨

<p><b>Build decisions, not just descriptions.</b></p>
<p>
  <a href="docs/index.zh-CN.md">完整文档</a> &nbsp;&nbsp;
  <a href="docs/docjev/architecture.md">工作原理</a> &nbsp;&nbsp;
  <a href="docs/docjev/training.md">训练方式</a> &nbsp;&nbsp;
  <a href="CONTRIBUTING.zh-CN.md">参与贡献</a>
</p>
<p>Apache-2.0（代码） &nbsp;&nbsp; <a href="NOTICE">NOTICE</a> &nbsp;&nbsp; <a href="THIRD_PARTY_LICENSES.zh-CN.md">第三方与模型许可证</a><br>感谢 <a href="https://github.com/Immortal-Zhang">Immortal-Zhang</a>、<a href="https://github.com/Kyousuke661">Kyousuke661</a> 与 <a href="https://github.com/BinyangQiu">BinyangQiu</a>。</p>
