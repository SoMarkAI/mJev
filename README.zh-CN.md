<p align="center">
  <img src="assets/mjev-cover-somark.svg" alt="mJev — Jev, with senses." width="100%">
</p>

<h1 align="center">mJev</h1>
<p align="center"><strong>简体中文</strong> · <a href="README.md">English</a></p>
<p align="center"><strong>Jev, with senses. mjev-doc, with documents.</strong></p>
<p align="center"><a href="https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD">🤗 mJev 权重</a> · <a href="#mjev-doc">📄 mjev-doc 模型说明</a></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modalities-Text%20%7C%20Vision-8b5cf6" alt="Text and Vision">
  <img src="https://img.shields.io/badge/Backends-HF%20%7C%20vLLM-0891b2" alt="HF and vLLM">
</p>

## 超越文本的决策智能

**Jev-style decision intelligence beyond text — multimodal state in, typed decisions out.**

mJev 将共享上下文转化为明确的选择：输入上下文，提出多个问题，并为每个问题定义候选项。每个结果包含最终选择、候选概率和原始 logits，便于检查决策并接入后续流程。训练模型聚焦通用视觉与文档理解；官方 Omni 后端支持音频、视频及带音轨视频。

- **共享上下文**：跨问题复用前缀 KV，支持问题批处理。
- **明确输出**：每道题拥有自己的候选集合，让每个决策对应一个预先定义的选项。
- **可运行流程**：在 HF／vLLM 上整合输入处理、候选评分和缓存一致性验证，默认从独立 HF 路径开始。

<a id="quick-start"></a>

## 🚀 快速开始

**mJev 与 mjev-doc 共用 HF 推理核心，统一以单张 24 GB NVIDIA GPU 为部署配置。**

需要 Linux、Python 3.11+ 和兼容的 NVIDIA 驱动。视频输入还需要系统 FFmpeg。

```bash
git clone https://gitlab.soulcode.cn/immortal/mjev.git
cd mjev
python3 -m venv .venv
source .venv/bin/activate
```

### 通用视觉 · mJev

```bash
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub

export MODEL_DIR="$HOME/models/mJev-Qwen3-VL-4B-RLCD"
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir "$MODEL_DIR"

# 运行仓库自带图片与两个问题
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model "$MODEL_DIR" \
  --input examples/multiple.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-demo.json
```

🎉 结果在 `outputs/image-demo.json`，示例采用 **HF + causal + stable**。

### 文档理解 · mjev-doc

使用完整的 mjev-doc checkpoint 目录，在同一环境中运行真实文档与多道中英文问题：

```bash
python -m pip install -e '.[docjev]'
CUDA_VISIBLE_DEVICES=0 python demo_docjev.py --model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --device-map cuda:0 \
  --output outputs/document-demo.json
```

结果在 `outputs/document-demo.json`，默认采用 **HF + causal + native BF16**。[模型权重与文档说明](#mjev-doc)

两个入口均无需安装 vLLM 或编译自定义 CUDA 内核。

GPU 数量由你选：用 `CUDA_VISIBLE_DEVICES` 指定可见设备。更多配置见 [HF 部署](docs/hf.zh-CN.md) · [vLLM 部署](docs/vllm.zh-CN.md)。

## 🧩 模型选择

| 模型 | 输入 | 从这里开始 |
| --- | --- | --- |
| **[mJev-Qwen3-VL-4B-RLCD](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD)** | 图片＋文本、视频＋文本 | [模型权重](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) · [安装、示例与评测](docs/models.zh-CN.md) |
| **mjev-doc** | 文档图片＋文本 | [模型介绍、示例与结果](#mjev-doc) |
| **Qwen3-Omni-30B-A3B-Instruct** | 图片、音频、视频、带音轨视频＋文本 | [Omni 部署教程](docs/hf.zh-CN.md) |

根据官方配置自动识别模型。图片／视频从 4B 开始；需要音频时切换到 Omni。两者使用相同的问题与候选项格式。

<a id="mjev-doc"></a>

## 📄 认识 mjev-doc

**mJev-Qwen3-VL-4B-RLCD** 面向通用视觉候选决策，**mjev-doc** 面向文档理解。两者共享 HF 推理核心，分别保留训练与评测说明；官方 Omni 后端继续支持音视频实验。

mjev-doc 使用 1,223 张文档、15,658 条中英文语言记录训练，冻结视觉参数、更新语言模型。首轮 500 条验证记录的准确率由 **73.0% 提升到 77.8%**。不同模型使用不同评测集，结果分别展示，不作直接排名。

```bash
python -m pip install -e '.[docjev]'
python demo_docjev.py --model /path/to/mjev-doc \
  --input examples/docjev/multiple.json --output outputs/docjev.json
```

mjev-doc 权重将与 mJev 一样通过 **Hugging Face** 发布，代码、模型介绍与评测统一维护在本仓库。当前示例支持完整 checkpoint 目录，也可使用官方 Qwen3-VL 基础模型。

评测包含 **1,512 道题目**。原始 Qwen 准确率 **82.04%**，mjev-doc **84.36%**（+2.31 个百分点）。每个候选均提供 raw logit 与 softmax 概率。[评测结果](docs/docjev/results.md)

[文档模型介绍](docs/docjev/README.zh-CN.md) · [训练](docs/docjev/training.md) · [评测结果](docs/docjev/results.md) · [真实验证集示例](examples/docjev/README.md)

## ⚡ 上下文不换，问题接着来

同一视频、16 个问题：已有受控实验中，前缀复用将整组耗时从 **32.62 秒降到 6.29 秒，提速 5.19×**。

[![HF cache latency comparison: ordinary batching versus prefix KV reuse](assets/cache-latency.svg)](docs/cache_scaling.zh-CN.md)

缓存也挑场合：长上下文、多问题更有用；单题可能更慢。长前缀 16 题的峰值已分配显存由 **15.80 增至 20.60 GiB**。这是单个自制视频的受控结果，不是所有任务的加速保证。

### 📏 性能测试详情

| 共享前缀 | 问题数 | 普通 batch | 前缀 KV 复用 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 63 tokens | 3 | 0.358 s | 0.341 s | 1.05× |
| 2,266 tokens | 1 | 2.467 s | 2.631 s | 0.94× |
| 2,266 tokens | 8 | 16.464 s | 4.246 s | **3.88×** |
| 2,266 tokens | 16 | 32.620 s | 6.287 s | **5.19×** |

条件：单张 24 GB NVIDIA GPU、Qwen3-VL-4B、HF `stable`（BF16 权重、FP32 文本计算）、`causal`、完整投影；预热两次后的五次均值，每次计时前清空分配器。包含媒体处理和首次前缀 prefill，不计模型加载。加速比为普通 batch 耗时除以缓存耗时，小于 1 表示更慢。

[全部八组配置与原始记录](docs/cache_scaling.zh-CN.md)

## 🎯 给它出点题

在 [mJev-Compositional-VQA](https://huggingface.co/datasets/Immortal-Zhang/mJev-Compositional-VQA) 的上游 **195 题历史评测集**上，经过 RLCD 训练，准确率从 **77.95% 提升至 80.00%（+2.05 个百分点）**，多答对 4 道题。

| 模型 | 正确 / 总题数 | 准确率 |
| --- | ---: | ---: |
| Qwen3-VL-4B-Instruct（训练前） | 152 / 195 | 77.95% |
| **[mJev-Qwen3-VL-4B-RLCD](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD)（训练后）** | **156 / 195** | **80.00%** |

两组模型使用相同的 195 道题进行评测，评测题独立于 RL 训练数据。准确率按正确题数除以总题数计算。公开数据集包含图片、问题、候选项与参考答案。

发布的模型使用 RLCD 针对候选选择进行训练，详见[训练与奖励设计](docs/training.zh-CN.md)。

## 📝 使用自己的数据

媒体路径相对于输入 JSON 文件解析。单问题示例见 [examples/single.json](examples/single.json)；多个问题共享媒体与上下文：

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

音视频使用 `modality` 和 `media_path`：`modality` 可选 `image`、`audio`、`video`、`audio_video`。视频应明确配置采样参数，完整示例见 [音视频输入说明](docs/hf.zh-CN.md#demo)。通用 mJev 示例使用项目自建媒体；mjev-doc 示例包含一张保留 CC-BY-2.0 署名的真实验证表格。

输出为 JSON 数组，每题一个结果，含 `candidates`、`decision` 和 `probability_sum`。可浏览[运行示例与已有输出记录](examples/README.zh-CN.md)。

实际结果还包含 token ID、并列决策信息、输入长度及缓存诊断。概率仅在本题候选集合内归一化，**不是校准后的置信度**。并列时选择输入顺序中的第一个候选。

候选数可变，接口边界为 2–128；所有标签必须在实际模板下通过单 token 校验，因此不保证每个数量都可用。空候选、重复候选和不合法控制 token 会被拒绝。HF 默认限制完整输入为 4000 tokens，超限报错，不静默截断。

## 🔎 工作方式

```text
媒体 + 公共上下文 → 官方 processor / chat template → 原生模型前缀 prefill
                                                       ├─ 问题 1 + candidates → Answer logits
                                                       ├─ 问题 2 + candidates → Answer logits
                                                       └─ 问题 N + candidates → Answer logits
候选标签 logits → 候选集合内 softmax → argmax 决策
```

多道题共享媒体，每道题有自己的候选集合。默认模式 `causal` 按顺序读取候选。实验模式 `isolated` 让每个候选读取共享上下文和问题，不读取其他候选；最终答案仍考虑全部候选。

模型原有输出层提供分数，mJev 返回所选答案与候选概率。HF／vLLM 的实现细节见[开发者指南](docs/development.zh-CN.md)。

## 📚 文档

| 你想做什么 | 去这里 |
| --- | --- |
| 部署、选模型 | [HF](docs/hf.zh-CN.md) · [vLLM](docs/vllm.zh-CN.md) · [模型选择](docs/models.zh-CN.md) |
| 跑文档任务、RLCD 或模型对比 | [mjev-doc](docs/docjev/README.zh-CN.md) · [评测结果](docs/docjev/results.md) |
| 找可运行示例 | [示例指南](examples/README.zh-CN.md) |
| 看代码、跑测试 | [开发者指南](docs/development.zh-CN.md) · [pytest](docs/testing.zh-CN.md) |
| 准备数据、复现评测 | [Benchmark](docs/benchmark.zh-CN.md) · [公开小型评测](docs/reproduce.zh-CN.md) |
| 查证据、看边界 | [最新验证](docs/validation_current.zh-CN.md) · [安装检查](docs/installation_validation.zh-CN.md) · [数值配置](docs/stability.zh-CN.md) |
| 探索实验功能 | [HTTP / Tree-KV](docs/integrated.zh-CN.md) · [音视频 Docker 镜像](experiments/av/README.zh-CN.md) |

## 🤝 贡献与交流

欢迎提交 bug、使用示例和改进建议。[提 Issue](https://gitlab.soulcode.cn/immortal/mjev/-/issues) · [发 PR](https://gitlab.soulcode.cn/immortal/mjev/-/merge_requests) · [贡献指南](CONTRIBUTING.zh-CN.md)。报告问题时，请附上环境、配置和最小复现步骤。

感谢 [Immortal-Zhang](https://github.com/Immortal-Zhang)、[Kyousuke661](https://github.com/Kyousuke661) 和 [BinyangQiu](https://github.com/BinyangQiu)。

## 📜 许可证

代码采用 [Apache-2.0](LICENSE)，上游署名见 [NOTICE](NOTICE)。官方模型权重单独下载，遵循各自许可证。第三方数据保留原始许可，**不随代码重授权**；详见 [第三方许可](THIRD_PARTY_LICENSES.zh-CN.md)。仓库不分发模型权重或完整第三方数据集。

mjev-doc 示例表格保留 CC-BY-2.0 许可及署名，见 [第三方许可](THIRD_PARTY_LICENSES.zh-CN.md)。
