[English](hf.md) · **简体中文**

[Qwen3-VL-4B 使用相同 HF API；安装方法及仅支持图片／视频的能力范围见模型指南。](models.zh-CN.md)

Docker 之外的命令均假定已激活 Python 3.11+ 虚拟环境：先用 `python3 -m venv .venv` 创建，再运行 `source .venv/bin/activate`。激活后安装和运行统一使用 `python`；Docker 内使用 `python3`。Shell 脚本也支持 `PYTHON=/path/to/venv/bin/python`。历史执行记录保留原始命令。

# Transformers 后端：无需 vLLM

> 历史结果按当时配置记录；后续检查与缺失的配置证据见[最新验证](validation_current.zh-CN.md)。

直接使用 `Qwen3OmniMoeThinkerForConditionalGeneration`，不加载 Talker、不调用 `generate()`、不添加可训练 head、不更新权重、不依赖 vLLM。SDPA 使用已安装的 PyTorch 内核，无需自定义编译。使用原样的本地官方 Qwen3-Omni 权重。

## 数值模式和 batch

默认 `--numerics stable`：官方 BF16 权重不变，文本激活、投影运算和路由使用 FP32；媒体编码不随问题 batch 大小改变。相比 `--numerics native` 会增加计算开销。native BF16 可复现早期结果，但不保证 batch/cache 一致性。比较时使用相同模式的串行基线。两种 HF 模式均不导入 vLLM，也不编译自定义 CUDA。范围和代价见 [稳定性文档](stability.zh-CN.md)。

```bash
python demo_hf.py --model /path/to/model --input task.json \
  --numerics stable --question-batch-size 3 --prefix-cache --projection full
```

## 安装

Linux、Python 3.11+（固定 PyAV 18.1.0 不支持 Python 3.10），视频音轨需要系统 FFmpeg；先安装适合硬件的 PyTorch 二进制包。从仓库根目录执行：

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf]'
```

核心固定版本为 Transformers 5.13.1、PyTorch 2.11.0 与 Torchvision 0.26.0。官方视频 processor 依赖 Torchvision，HF extra 会显式安装。

HF extra 还固定与 PyTorch 2.11 匹配的 `torchcodec==0.11.0+cpu`，由官方 Qwen 解码路径读取 Decord 无法处理的公开 mini AV1 视频。公开 HF 评测显式设置 `FORCE_QWENVL_VIDEO_READER=torchcodec`，需要系统 FFmpeg 共享库。先按上方命令从官方 PyTorch 索引安装 CPU 解码包，避免额外 CUDA 视频库依赖；模型推理仍使用 GPU。后续 PyTorch/HF 安装不要沿用 CPU 索引。版本对应关系见 [官方兼容表](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions)。

vLLM 为可选 extra：`pip install -e '.[vllm]'`，仅用于 vLLM 路径。HF 不设置 `MJEV_ENABLE` 或 `MJEV_ENABLE_PATCHES`，也不需要或读取已安装的 vLLM。

<a id="demo"></a>

## Demo

```bash
python demo_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --input examples/single.json --check-only
python demo_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --input examples/single.json --mode causal --output outputs/hf-causal.json
python demo_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --input examples/single.json --mode isolated --output outputs/hf-isolated.json
```

音视频输入示例：

```json
{
  "modality": "audio_video",
  "media_path": "clip.mp4",
  "video_options": {"fps": 1, "min_pixels": 3136, "max_pixels": 50176, "max_frames": 32},
  "questions": [
    {"id": "q1", "question": "Is anyone speaking?", "candidates": ["yes", "no"]},
    {"id": "q2", "question": "What is moving?", "candidates": ["a person", "a vehicle", "neither"]}
  ]
}
```

`modality` 为 image、audio、video 或 audio_video，路径相对 JSON 文件。上述参数从完整视频采样帧，不按参考答案证据裁剪。显式配置便于复现。`--check-only` 仅加载官方 processor/tokenizer，检查完整实际输入长度，默认 4000；超限报错，不静默截断。处理参数改变后，旧 benchmark token 数不一定适用。

输出沿用主包字段：候选 label/text/token_id/raw_logit/probability、确定性 decision、并列元信息和实际 prompt 长度。softmax 仅在给定候选间进行，不是校准置信度。实际原生 `Answer:` 后验证标签为单 token；重复选项与主协议一致地拒绝。

## 实现与边界

官方预处理和 MRoPE 在原始 2D attention mask 上运行；临时 hook 在 MRoPE 计算完成后的 Thinker 文本模型边界注入 4D 加性可见性 mask。mask 遵循所选模式：`causal` 保留普通因果可见性；`isolated` 让候选看公共前缀及自身历史。决策后缀看全部候选。此自定义 mask 路径仅支持 eager/SDPA 文本 attention，不支持 FlashAttention。

只读取最后位置的原 LM Head。默认 `--projection selected` 用原始权重的选定行进行 FP32 乘法；`--projection full` 在最后位置算完整 head 后选标签。BF16 投影舍入可能不同，该选项用于诊断，不保证所有设备逐位一致。异常时也会移除 hooks，同一 engine 的调用串行执行。

多卡使用 `device_map=auto`，不是 vLLM TP。包装器将官方文本 decoder layer 类加入 `_no_split_modules`，避免 Transformers 5.13.1 将残差块拆到不同设备；仅改变放置，不改变 forward 数学或权重。Python 可传 `max_memory={0: "20GiB", 1: "20GiB", 2: "20GiB", 3: "20GiB"}` 预留激活/缓存空间，按设备调整。stable 自动放置在未指定预算时，每张可见 GPU 预留 4 GiB。量化和磁盘 offload 未验证。

默认多问题复用已解码媒体，但逐题重算完整输入。demo/评估器加 `--prefix-cache` 后仅 prefill 一次媒体和公共上下文，每题使用独立 KV 副本计算后缀。causal/isolated 都支持。`--question-batch-size 3` 是真实三行 forward；Python 的 `score_many`/`score_questions` 使用 `batch_size=3`。同一 engine 的调用/组仍串行，组内问题 batch。缓存是每媒体显式复用，不是全局自动缓存，也不声称具备 vLLM scheduler 或 Tree-KV。HF 脚本不会停止 GPU 服务。

## 显式前缀缓存接口

```python
from mjev.hf import HFMJevEngine
engine = HFMJevEngine("/path/to/Qwen3-Omni-30B-A3B-Instruct")
context = engine.prepare_context("clip.mp4", modality="audio_video",
    video_options={"fps": 1, "min_pixels": 3136, "max_pixels": 50176, "max_frames": 32})
results = engine.score_questions(context, [
    {"id": "q1", "question": "Is anyone speaking?", "candidates": ["yes", "no"]},
    {"id": "q2", "question": "Is music audible?", "candidates": ["yes", "no"]},
], mode="isolated", batch_size=3)
del context  # release retained media features and prefix KV
```

JSON demo 可用 `python demo_hf.py --model /path/to/model --input task.json --mode isolated --prefix-cache`。评估器接受同一开关；便利 API 为 `score_many(..., use_prefix_cache=True)`。`num_cached_tokens` 报告实际复用公共前缀长度。

边界由官方 tokenizer offsets 在问题文本之前计算，跨边界的 BPE token 保留于后缀。每个完整输入计算原生 MRoPE，再与缓存前缀核对。复用时检查媒体特征、时间元数据和前缀 tokens，候选 mask 正确计入缓存 key 偏移。原媒体编码器仅在 prefill 运行，但 processor 仍对保留的已解码媒体逐题工作。

context 为 engine 内存中的不透明对象；保留期间不可修改字段或模型权重/device/dtype。媒体、采样、prompt 或模型改变时重建。没有磁盘持久化、淘汰策略或零复制 Tree-KV。基础 KV 与各 batch 分支增长的副本共存，用显存/带宽换分支隔离。短输入或单题不一定受益，真实速度/显存需 GPU 测量。完整输入仍须满足长度限制。

## CPU 测试

```bash
pip install pytest
python -m pytest -q tests/hf
```

使用真实的小型随机权重 Qwen3-Omni Thinker forward，覆盖图像/音频/视频、带 causal 正对照的隔离干预、selected/full 投影、概率、权重不变、hook 清理、缓存 logits 一致性、编码器绕过、问题顺序、独立 KV 分支与过期前缀拒绝。这是实现证据，不是 30B 模型评测。

早期 25 项 HF 测试、59 项仓库测试通过均为历史数；当前扩大后的验证见 [稳定性记录](stability.zh-CN.md)。另一次 CPU 检查将官方 processor、真实 Clotho-AQA/MuChoMusic/Video-MME-v2/MMOU 媒体配合含 DeepStack 的 tiny 随机 Thinker，每媒体两题、两模式，缓存/未缓存最大 logit 绝对差 5.96e-8。这不代表准确率或完整 BF16 模型一致性。官方 processor、GPU 和全权重验证是不同层次。

全权重无缓存配对评测已完成 1,249 道图片/音频/视频题，见 [配对结果](paired_backends.zh-CN.md)。持续并发吞吐和后端精确数值相等未建立。下面历史 native BF16 缓存验收失败，stable 回归另行记录。

## 历史 native BF16：缓存验收失败

2026-09-26 修正运行在四张 24 GB NVIDIA GPU 加载官方 native BF16 Thinker，完整 decoder layer 放置，selected-logit SDPA 评分。无缺失/不匹配 Thinker key；未用 Talker/Code2Wav 权重忽略。抽样对照 checkpoint 的 288 个完整 expert 张量（48 层各两个 expert、三种张量）全部一致，并保存元信息与哈希；不是对全部 checkpoint 张量逐一校验。

固定四个媒体、八道原标签题，causal/isolated 两模式。原始缓存/未缓存答案 14/16 一致：causal 8/8，isolated 6/8；分歧为一道 Video-MME-v2 和一道 MMOU。另 24 项问题顺序和排列输入下的缓存/未缓存答案比较均一致；问题顺序 logits 完全相同，但八个候选排列对照全部未通过 raw-logit 容差，三项也未通过概率容差。这不意味着改变选项标签后模型不变。共 40 项，最大 logit 差 1.126003、概率差 0.126645，超过预设 0.1/0.02。运行完成但验收失败，保留 `ERROR.json`，无成功标记。

该历史 native 全权重差异原因未解决，不能仅归因 BF16 舍入。native prefix cache 仍有警告。新 stable 配置的独立验证见 [稳定性文档](stability.zh-CN.md)。参考评分使用未缓存路径；不能由该失败试验声称缓存等价、答案不变、准确率或生产可用。

两题的本地预热耗时，包含预处理/缓存构建，不含加载，每格一次，不是 HTTP 延迟：

| 样本 | 模式 | 无 KV 复用 | 有 KV 复用 |
| --- | --- | ---: | ---: |
| Clotho-AQA | causal / isolated | 0.642 / 0.703 秒 | 0.816 / 0.803 秒 |
| MuChoMusic | causal / isolated | 0.638 / 0.655 秒 | 0.825 / 0.820 秒 |
| Video-MME-v2 | causal / isolated | 7.059 / 7.634 秒 | 6.741 / 6.779 秒 |
| MMOU | causal / isolated | 1.353 / 1.454 秒 | 1.443 / 1.396 秒 |

答案不一致且样本极小，不能证明保正确性的加速。每 GPU 的 PyTorch 峰值 allocated 约为 15.78、16.78、16.78、12.69 GiB，不等于驱动/预留显存。

更早 v2 使用子类绕过原生 MoE 权重转换，expert 被重新初始化，因此那次所有官方模型结论无效，仅保留诊断。现在使用原生类，对缺失/不匹配 Thinker 权重或非 Talker/Code2Wav 的意外 key 中止。回归使用未融合逐 expert checkpoint，核对 tiny 模型全部张量并拒绝不完整 checkpoint。

在四张空闲 GPU、新输出目录复现：

```bash
python benchmarks/integrated/validate_hf_gpu.py --model /path/to/model --root /path/to/benchmark --out outputs/hf-gpu-check --numerics native --model-revision "$MODEL_REVISION"
```

保存加载信息、源码哈希、抽样 checkpoint 见证、冻结 manifest、原分数、耗时与验收统计；容差失败保留结果并停止，不放宽门槛。

## 数据集评估器

```bash
pip install -e '.[hf,benchmark]'
python benchmarks/integrated/evaluate_hf.py --model /path/to/Qwen3-Omni-30B-A3B-Instruct --root /path/to/benchmark --out outputs/hf-causal --mode causal --numerics stable --model-revision "$MODEL_REVISION" --projection selected --question-batch-size 1
```

每次用新输出目录，复用主包 Accuracy/NLL/Brier/ECE，沿用集成 vLLM 的原标签 MCQ 筛选。Clotho 的 374 道原生/分歧题明确排除，不重生成问题与参考。错误停止并保留 `ERROR.json` 和已完成预测。视频参数显式配置；qwen-omni-utils 完成缩放后禁用 HF 二次视频缩放，避免套用不相关默认值。这是可复现的新 HF 采样配置，不证明与旧 vLLM token 长度相同。

> 命令中的 MODEL_REVISION 必须设为该本地模型真实对应的 40 字符 commit。这里的 native 命令是显式选择的重跑配置，不是对历史配置缺失项的补填，也不保证重现旧成绩。公开可下载流程请用 [公开小型评测](reproduce.zh-CN.md)。
