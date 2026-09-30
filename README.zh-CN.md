<p align="center">
  <img src="assets/mjev-cover-somark.svg" alt="mJev — Jev, with senses." width="100%">
</p>

<h1 align="center">mJev</h1>
<p align="center"><strong>简体中文</strong> · <a href="README.md">English</a></p>
<p align="center"><strong>Jev, with senses.</strong></p>
<p align="center"><a href="https://huggingface.co/SoMarkAI/mJev">🤗 Hugging Face 模型</a></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modalities-Text%20%7C%20Vision-8b5cf6" alt="Text and Vision">
  <img src="https://img.shields.io/badge/Backends-HF%20%7C%20vLLM-0891b2" alt="HF and vLLM">
</p>

## 超越文本的决策智能

**Jev-style decision intelligence beyond text — multimodal state in, typed decisions out.**

Text and vision today. Audio, video, and more to come.

mJev 将共享上下文转化为明确的选择：输入上下文，提出多个问题，并为每个问题定义候选项。每个结果包含最终选择、候选概率和原始 logits，便于检查决策并接入后续流程。当前聚焦文本与视觉，未来扩展到音频、视频及更多模态。

- **共享上下文**：跨问题复用前缀 KV，支持问题批处理。
- **明确输出**：每道题拥有自己的候选集合，让每个决策对应一个预先定义的选项。
- **可运行流程**：在 HF／vLLM 上整合输入处理、候选评分和缓存一致性验证，默认从独立 HF 路径开始。

## 🎬 六秒小剧场

[![动态视频预览：红球先向右移动，随后蓝色方块上升](examples/motion-demo/preview.gif)](examples/motion-demo/motion.mp4)

**[打开 6 秒视频](examples/motion-demo/motion.mp4)** · [三题输入](examples/motion-demo/input.json) · [完整实测结果](examples/motion-demo/recorded-output.json)

红球先走，蓝方块接棒。六秒小剧场，看看模型有没有跟上 👀

下表为 **Qwen3-VL-4B + HF 的实际输出**，使用项目自制动画。

| 问题 | 候选项概率 | 最终选择 |
| --- | --- | --- |
| How does the red ball move? | From left to right: 71.34%<br>From right to left: 28.37%<br>Upward: 0.27%<br>It stays still: 0.02% | **From left to right** |
| Which object moves upward? | The red ball: 2.33%<br>The blue square: 97.29%<br>Both objects: 0.38% | **The blue square** |
| Which movement happens first? | The red ball moves right: 89.62%<br>The blue square moves up: 10.04%<br>Both movements begin at the same time: 0.34% | **The red ball moves right** |

**候选内相对概率，不是校准置信度。** 数字经过四舍五入。

<a id="quick-start"></a>

## 🚀 跑起来，轮到你了

**已在单张 24 GB NVIDIA GPU 上验证。** 需要 Linux、Python 3.11+、兼容的 NVIDIA 驱动和系统 FFmpeg。

```bash
git clone https://github.com/SoMarkAI/mJev.git
cd mJev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub

export MODEL_DIR="$HOME/models/mJev"
hf download SoMarkAI/mJev --local-dir "$MODEL_DIR"

# 运行上面的同一段视频与三个问题
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model "$MODEL_DIR" \
  --input examples/motion-demo/input.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 3 --output outputs/motion-demo.json
```

🎉 结果在 `outputs/motion-demo.json`。默认走 **HF + causal + stable**，无需安装 vLLM 或编译自定义 CUDA 内核。

GPU 数量由你选：用 `CUDA_VISIBLE_DEVICES` 指定可见设备。更多配置见 [HF 部署](docs/hf.zh-CN.md) · [vLLM 部署](docs/vllm.zh-CN.md)。

## 🧩 选个搭档

| 模型 | 输入 | 从这里开始 |
| --- | --- | --- |
| **[mJev（Qwen3-VL-4B）](https://huggingface.co/SoMarkAI/mJev)** | 图片＋文本、视频＋文本 | [模型权重](https://huggingface.co/SoMarkAI/mJev) · [安装、示例与评测](docs/models.zh-CN.md) |
| **Qwen3-Omni-30B-A3B-Instruct** | 图片、音频、视频、带音轨视频＋文本 | [Omni 部署教程](docs/hf.zh-CN.md) |

根据官方配置自动识别模型。图片／视频从 4B 开始；需要音频时切换到 Omni。两者使用相同的问题与候选项格式。

## 🔎 好奇里面怎么转？

```text
媒体 + 公共上下文 → 官方 processor / chat template → 原生模型前缀 prefill
                                                       ├─ 问题 1 + candidates → Answer logits
                                                       ├─ 问题 2 + candidates → Answer logits
                                                       └─ 问题 N + candidates → Answer logits
候选标签 logits → 候选集合内 softmax → argmax 决策
```

每道题，都带着自己的选项小队 🧩

- **默认 `causal`**：选项按顺序入场，后面的可以读到前面的内容。
- **想做对照实验？试试 `isolated`**：给每个选项一间“小隔间”，共享上下文和问题，互不偷看；最后的决策仍会综合所有选项的信息。

评分交给模型原有的输出层，你拿到选择和候选概率。好奇 HF 与 vLLM 怎么实现？[技术细节在这里](docs/development.zh-CN.md)。

## 训练与奖励设计

发布的 mJev 模型使用 [GRPO](https://arxiv.org/abs/2402.03300) 针对候选选择进行微调。训练前，冻结的评分器为每条样本计算目标概率 $p^{\ast}$，即正确选项在候选集合内的概率。每次采样的回答按下式获得奖励：

```math
r(\hat{y}) = \begin{cases}
+p^{\ast}, & \hat{y} = y^{\ast} \\
-p^{\ast}, & \text{otherwise}
\end{cases}
```

其中 $y^{\ast}$ 为正确标签，错误或无效回答获得负奖励。奖励范围为 $[-1, 1]$，目标概率越高，训练信号越强。输出被限制为单个候选标签，因此不另加格式奖励；关闭 reward scaling 以保留这一权重，并通过独立的 KL 惩罚限制策略偏离参考模型。训练更新语言模型，视觉塔和对齐模块保持冻结。

## ⚡ 上下文不换，问题接着来

同一视频、16 个问题：已有受控实验中，前缀复用将整组耗时从 **32.62 秒降到 6.29 秒，提速 5.19×**。

[![HF cache latency comparison: ordinary batching versus prefix KV reuse](assets/cache-latency.svg)](docs/cache_scaling.zh-CN.md)

缓存也挑场合：长上下文、多问题更有用；单题可能更慢。长前缀 16 题的峰值已分配显存由 **15.80 增至 20.60 GiB**。这是单个自制视频的受控结果，不是所有任务的加速保证。

### 📏 完整数字与测量条件

| 共享前缀 | 问题数 | 普通 batch | 前缀 KV 复用 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 63 tokens | 3 | 0.358 s | 0.341 s | 1.05× |
| 2,266 tokens | 1 | 2.467 s | 2.631 s | 0.94× |
| 2,266 tokens | 8 | 16.464 s | 4.246 s | **3.88×** |
| 2,266 tokens | 16 | 32.620 s | 6.287 s | **5.19×** |

条件：单张 24 GB NVIDIA GPU、Qwen3-VL-4B、HF `stable`（BF16 权重、FP32 文本计算）、`causal`、完整投影；预热两次后的五次均值，每次计时前清空分配器。包含媒体处理和首次前缀 prefill，不计模型加载。加速比为普通 batch 耗时除以缓存耗时，小于 1 表示更慢。

[全部八组配置与原始记录](docs/cache_scaling.zh-CN.md)

## 🎯 给它出点题

在 [mJev-Compositional-VQA](https://huggingface.co/datasets/Immortal-Zhang/mJev-Compositional-VQA) 的 **195 题评测子集**上，经过 GRPO 微调，准确率从 **77.95% 提升至 80.00%（+2.05 个百分点）**，多答对 4 道题。

| 模型 | 正确 / 总题数 | 准确率 |
| --- | ---: | ---: |
| Qwen3-VL-4B-Instruct（GRPO 训练前） | 152 / 195 | 77.95% |
| **[mJev](https://huggingface.co/SoMarkAI/mJev)（GRPO 训练后）** | **156 / 195** | **80.00%** |

两组模型使用相同的 195 道题进行评测，评测题独立于 RL 训练数据。准确率按正确题数除以总题数计算。公开数据集包含图片、问题、候选项与参考答案。

## 📝 换成自己的输入

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

音视频使用 `modality` 和 `media_path`：`modality` 可选 `image`、`audio`、`video`、`audio_video`。视频应明确配置采样参数，完整示例见 [音视频输入说明](docs/hf.zh-CN.md#demo)。仓库示例使用合成图片，不包含第三方评测媒体。

输出为 JSON 数组，每题一个结果，含 `candidates`、`decision` 和 `probability_sum`。可查看上方 [实际输出记录](examples/motion-demo/recorded-output.json) 的 `questions` 字段。

实际结果还包含 token ID、并列决策信息、输入长度及缓存诊断。概率仅在本题候选集合内归一化，**不是校准后的置信度**。并列时选择输入顺序中的第一个候选。

候选数可变，接口边界为 2–128；所有标签必须在实际模板下通过单 token 校验，因此不保证每个数量都可用。空候选、重复候选和不合法控制 token 会被拒绝。HF 默认限制完整输入为 4000 tokens，超限报错，不静默截断。

## 📚 按需翻阅，不用从头啃

| 你想做什么 | 去这里 |
| --- | --- |
| 部署、选模型 | [HF](docs/hf.zh-CN.md) · [vLLM](docs/vllm.zh-CN.md) · [模型选择](docs/models.zh-CN.md) |
| 看代码、跑测试 | [开发者指南](docs/development.zh-CN.md) · [pytest](docs/testing.zh-CN.md) |
| 准备数据、复现评测 | [Benchmark](docs/benchmark.zh-CN.md) · [公开小型评测](docs/reproduce.zh-CN.md) |
| 查证据、看边界 | [最新验证](docs/validation_current.zh-CN.md) · [安装检查](docs/installation_validation.zh-CN.md) · [数值配置](docs/stability.zh-CN.md) |
| 探索实验功能 | [HTTP / Tree-KV](docs/integrated.zh-CN.md) · [音视频 Docker 镜像](experiments/av/README.zh-CN.md) |

## 🤝 来，一起添块积木

发现 bug、想到新场景、想改两行文档？都欢迎！[提 Issue](https://github.com/SoMarkAI/mJev/issues) · [发 PR](https://github.com/SoMarkAI/mJev/pulls) · [贡献指南](CONTRIBUTING.zh-CN.md)。报告问题时，记得带上环境、配置和最小复现步骤。

感谢 [Immortal-Zhang](https://github.com/Immortal-Zhang)、[Kyousuke661](https://github.com/Kyousuke661)、[BinyangQiu](https://github.com/BinyangQiu)，也期待你的名字出现在这里 ✨

## 📜 许可证

代码采用 [Apache-2.0](LICENSE)，上游署名见 [NOTICE](NOTICE)。官方模型权重单独下载，遵循各自许可证。第三方数据保留原始许可，**不随代码重授权**；详见 [第三方许可](THIRD_PARTY_LICENSES.zh-CN.md)。仓库不分发模型权重和第三方评测媒体。
