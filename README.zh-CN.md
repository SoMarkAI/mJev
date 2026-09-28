<p align="center">
  <img src="assets/mjev-banner.svg" alt="mJev — Jev, with senses." width="100%">
</p>

<h1 align="center">mJev</h1>
<p align="center"><strong>简体中文</strong> · <a href="README.md">English</a></p>
<p align="center"><strong>Jev, with senses.</strong></p>
<p align="center">
  <a href="#quick-start">快速开始</a> ·
  <a href="#performance">性能</a> ·
  <a href="#evaluation">评测</a> ·
  <a href="#documentation">文档</a> ·
  <a href="docs/validation_current.zh-CN.md">最新验证</a> ·
  <a href="CONTRIBUTING.zh-CN.md">参与贡献</a>
</p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modalities-Image%20%7C%20Video%20%7C%20Audio-8b5cf6" alt="Image, Video and Audio">
  <img src="https://img.shields.io/badge/Backends-HF%20%7C%20vLLM-0891b2" alt="HF and vLLM">
</p>

## 将共享上下文转化为明确决策

一段视频可以引出不止一个问题：什么在移动、哪个对象值得关注、哪个动作先发生？mJev 将这些问题整合进同一套流程。提供媒体，为每道题定义候选项，即可得到最终选择和各候选项的概率。

面向**视频属性标注、固定选项视觉问答与多模态评测**，mJev 在 HF 和 vLLM 之上串联媒体处理、问题分支、候选评分与缓存一致性检查。

- **同一份上下文，回答多个问题。** 跨问题复用媒体前缀，并批量执行问题分支。
- **由你定义选项，输出结构化结果。** 每道题拥有独立的候选集合，结果可直接接入后续流程。
- **从小模型起步，扩展到多种模态。** 使用 Qwen3-VL-4B 处理图片和视频，或使用 Qwen3-Omni 处理音频与带音轨视频。

## 看一次实际输出

[![动态视频预览：红球先向右移动，随后蓝色方块上升](examples/motion-demo/preview.gif)](examples/motion-demo/motion.mp4)

**[打开 6 秒视频](examples/motion-demo/motion.mp4)** · [三题输入](examples/motion-demo/input.json) · [完整实测结果](examples/motion-demo/recorded-output.json)

红球先向右移动，蓝色方块随后上升。同一段视频，三个问题：发生了什么、谁在移动、哪个动作在先？下面是 **Qwen3-VL-4B 的实际输出**。

| 问题 | 候选项概率 | 最终选择 |
| --- | --- | --- |
| How does the red ball move? | From left to right: 71.34%<br>From right to left: 28.37%<br>Upward: 0.27%<br>It stays still: 0.02% | **From left to right** |
| Which object moves upward? | The red ball: 2.33%<br>The blue square: 97.29%<br>Both objects: 0.38% | **The blue square** |
| Which movement happens first? | The red ball moves right: 89.62%<br>The blue square moves up: 10.04%<br>Both movements begin at the same time: 0.34% | **The red ball moves right** |

**候选内相对概率，不是校准置信度。** 数字经过四舍五入。

项目自制动画，HF 单卡实测；配置与原始结果见上方记录。

<a id="quick-start"></a>

## 快速开始

**已在单张 24 GB NVIDIA GPU 上验证，可运行下方示例。** 准备 Linux、Python 3.11+、兼容的 NVIDIA 驱动和系统 FFmpeg。

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

# 运行上面的同一段视频与三个问题
CUDA_VISIBLE_DEVICES=0 python demo_hf.py --model "$MODEL_DIR" \
  --input examples/motion-demo/input.json --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 3 --output outputs/motion-demo.json
```

默认入口是 **HF**（`demo_hf.py`），使用 `causal` 注意力和 `stable` 数值配置。运行此示例无需 Docker、vLLM 或编译自定义 CUDA 内核。结果保存至 `outputs/motion-demo.json`。

通过 `CUDA_VISIBLE_DEVICES` 指定 HF 可用的一张或多张 GPU；实际容量取决于模型、输入长度和问题批大小。更多设置见 [HF 部署](docs/hf.zh-CN.md)，可选后端见 [vLLM 部署](docs/vllm.zh-CN.md)。

<a id="performance"></a>

## 什么时候前缀复用更划算

在已记录的 16 题视频受控实验中，前缀复用将整组耗时从 **32.62 秒降至 6.29 秒**，已包含首次构建前缀的时间。收益取决于这些问题共享多少上下文：

| 共享前缀 | 问题数 | 普通 batch | 前缀 KV 复用 | 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 63 tokens | 3 | 0.358 s | 0.341 s | 1.05× |
| 2,266 tokens | 1 | 2.467 s | 2.631 s | 0.94× |
| 2,266 tokens | 8 | 16.464 s | 4.246 s | **3.88×** |
| 2,266 tokens | 16 | 32.620 s | 6.287 s | **5.19×** |

<details>
<summary>测量条件与显存代价</summary>

条件：单张 24 GB NVIDIA GPU、Qwen3-VL-4B、HF `stable`（BF16 权重、FP32 文本计算）、`causal`、完整投影；预热两次后的五次均值，每次计时前清空分配器。包含媒体处理和首次前缀 prefill，不计模型加载。加速比为普通 batch 耗时除以缓存耗时，小于 1 表示更慢。

长前缀 16 题的峰值已分配显存从 **15.80 增至 20.60 GiB**，包含模型权重。这是一个项目自有合成视频上的受控结果，不代表所有任务均能获得相同加速。[全部八组配置、原始结果与复现方法](docs/cache_scaling.zh-CN.md)

</details>

<a id="evaluation"></a>

## 评测

**[mJev-Compositional-VQA](https://huggingface.co/datasets/Immortal-Zhang/mJev-Compositional-VQA)** 提供用于组合式视觉推理的图片问题、候选项与参考答案。数据集独立托管于 Hugging Face，获取方式、格式和逐图许可证请查看数据集说明。该 benchmark 的评测结果将在发布后于此链接。

## 模型与后端

| 模型 | 输入 | 从这里开始 |
| --- | --- | --- |
| **Qwen3-VL-4B-Instruct** | 图片＋文本、视频＋文本 | [4B 安装、示例与评测](docs/models.zh-CN.md) |
| **Qwen3-Omni-30B-A3B-Instruct** | 图片、音频、视频、带音轨视频＋文本 | [Omni 部署教程](docs/hf.zh-CN.md) |

根据官方配置自动识别模型。图片／视频从 4B 开始；需要音频时切换到 Omni。两者使用相同的问题与候选项格式。

HF 直接调用模型 `forward`。可选的 vLLM pooling 后端使用 `AsyncLLM.encode`，依赖固定版本与自定义 hooks；两条主路径均不调用 `generate()`。HTTP 与 Tree-KV 属于[实验路径](docs/integrated.zh-CN.md)，验证范围单独说明。

<details>
<summary><strong>输入与输出格式</strong></summary>

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

音视频使用 `modality` 和 `media_path`：`modality` 可选 `image`、`audio`、`video`、`audio_video`。视频应明确配置采样参数，完整示例见 [音视频输入说明](docs/hf.zh-CN.md#demo)。仓库示例使用项目自制媒体，第三方评测媒体单独托管。

输出为 JSON 数组，每题一个结果，含 `candidates`、`decision` 和 `probability_sum`。可查看上方 [实际输出记录](examples/motion-demo/recorded-output.json) 的 `questions` 字段。

实际结果还包含 token ID、并列决策信息、输入长度及缓存诊断。概率仅在本题候选集合内归一化，**不是校准后的置信度**。并列时选择输入顺序中的第一个候选。

候选数可变，接口边界为 2–128；所有标签必须在实际模板下通过单 token 校验，因此不保证每个数量都可用。空候选、重复候选和不合法控制 token 会被拒绝。HF 默认限制完整输入为 4000 tokens，超限报错，不静默截断。

</details>

## 推理过程

```text
媒体 + 公共上下文 → 官方 processor / chat template → 原生模型前缀 prefill
                                                       ├─ 问题 1 + candidates → Answer logits
                                                       ├─ 问题 2 + candidates → Answer logits
                                                       └─ 问题 N + candidates → Answer logits
候选标签 logits → 候选集合内 softmax → argmax 决策
```

每个问题拥有自己的候选集合。`causal` 使用普通因果可见性；`isolated` 让每个候选只能看到公共上下文、问题和自身的历史 token，最终 `Answer:` 位置可以看到全部候选。默认使用 `causal`；`isolated` 作为可控的注意力实验选项。

各问题分支相互独立。候选隔离是**同一道题内部**的可选设置，前缀复用不要求开启候选隔离。HF 将前缀 KV 复制到问题分支；`stable` 数值配置会增加计算与显存开销，详情见[数值配置](docs/stability.zh-CN.md)。

<a id="documentation"></a>

## 文档导航

| 我想要…… | 从这里开始 |
| --- | --- |
| 部署模型或使用 Python API | [模型选择](docs/models.zh-CN.md) · [HF](docs/hf.zh-CN.md) · [vLLM](docs/vllm.zh-CN.md) |
| 理解评分、缓存与数值配置 | [开发者指南](docs/development.zh-CN.md) · [稳定性](docs/stability.zh-CN.md) |
| 准备数据并复现评测 | [Benchmark](docs/benchmark.zh-CN.md) · [公开小型评测](docs/reproduce.zh-CN.md) |
| 了解已经验证的内容 | [最新验证](docs/validation_current.zh-CN.md) · [安装检查](docs/installation_validation.zh-CN.md) |
| 贡献代码或运行测试 | [贡献指南](CONTRIBUTING.zh-CN.md) · [pytest 测试范围](docs/testing.zh-CN.md) |
| 探索可选 runtime 实现 | [HTTP / Tree-KV](docs/integrated.zh-CN.md) · [音视频 Docker 镜像](experiments/av/README.zh-CN.md) |

## 社区与贡献

欢迎分享可复现的问题、新的使用场景、文档改进与受控对照实验。阅读[贡献指南](CONTRIBUTING.zh-CN.md)，提交 [Issue](https://github.com/SoMarkAI/mJev/issues)，或发起 [Pull Request](https://github.com/SoMarkAI/mJev/pulls)。

报告问题时请附上 commit、环境、后端、数值配置和最小复现步骤。请勿提交凭据、私有媒体或模型权重。

感谢贡献者：[Immortal-Zhang](https://github.com/Immortal-Zhang)、[Kyousuke661](https://github.com/Kyousuke661) 和 [BinyangQiu](https://github.com/BinyangQiu)。

## 许可证

mJev 代码使用 [Apache License 2.0](LICENSE)，上游署名见 [NOTICE](NOTICE)。官方 [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 和 [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) 权重单独下载并遵循各自许可证。

第三方数据的标注、音频和视频保留原始权利与限制，**不会被本项目统一重授权为 Apache-2.0**。详情见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.zh-CN.md)。仓库不分发模型权重和 benchmark 媒体。
