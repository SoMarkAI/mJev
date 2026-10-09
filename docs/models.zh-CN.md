[English](models.md) · **简体中文**

[官方模型历史验证记录](qwen3_vl_validation.zh-CN.md)

# 模型选择与部署

mJev 根据本地 `config.json` 自动识别模型家族。图片／视频从已发布的 [mJev-Qwen3-VL-4B-RLCD](https://huggingface.co/SoMarkAI/mJev-Qwen3-VL-4B-RLCD) 开始，需要音频时选择官方 Qwen3-Omni Thinker 权重。两条路径保留原生 processor、chat template、视觉特征、位置编码与 LM Head，通过 `forward` 或 vLLM pooling 读取现有输出层进行评分。

| 能力 | Qwen3-Omni-30B-A3B-Instruct | Qwen3-VL 家族（4B） |
| --- | --- | --- |
| 图片＋文本 | 支持 | 支持 |
| 视频＋文本 | 支持 | 支持 |
| 音频／视频音轨 | 支持 | **不支持，显式拒绝** |
| 动态候选、logits、概率与决策 | 统一接口 | 统一接口 |
| HF 因果／候选隔离 attention | 支持 | 支持 |
| HF 公共前缀 KV 与真实问题批处理 | 支持 | 支持 |
| vLLM pooling、mask、prefix cache | 可选后端 | 可选后端 |
| 默认 vLLM 张量并行度 | 4 卡 | 1 卡 |
| 权重 | 单独下载官方权重 | 单独下载已发布 RLCD 或官方基础权重 |

已发布的 RLCD 模型针对候选选择进行了微调，详见[训练与奖励设计](training.zh-CN.md)。归档的 GPU 验证记录与固定评测配置使用 **Qwen/Qwen3-VL-4B-Instruct**，每份结果应保留实际使用的模型身份。

Qwen3-VL 是稠密视觉语言模型，参数较少不等于已经证明某个加速倍数。它不能替代 Omni 处理音频任务；尤其不能去掉音频相关 benchmark 的音轨后，将成绩当作同等条件的评测。

## 安装与运行 4B

文档中的 GPU／视频流程需要 Linux、Python 3.11+、NVIDIA GPU 和系统 FFmpeg。HF 无需安装 vLLM 或编译自定义 CUDA 内核；CPU codec 负责视频解码，CUDA PyTorch 负责模型推理。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl,test]' huggingface_hub
export MODEL_DIR="$HOME/models/mJev-Qwen3-VL-4B-RLCD"
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir "$MODEL_DIR"
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json --check-only
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json \
  --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/vl-image.json
```

使用与 [HF 部署](hf.zh-CN.md) 相同的 `HFMJevEngine` API 和输入 JSON。模型类型由配置识别，不接受未经核对的 CLI 强行覆盖。VL 在解码前拒绝 `audio` 与 `audio_video`，视频保留原生时间戳与 `mm_token_type_ids`，不替换位置编码算法。

可选的固定版本 vLLM 环境按 [部署文档](vllm.zh-CN.md) 准备，从仓库目录运行：

```bash
VLLM_BATCH_INVARIANT=1 python demo.py --model "$MODEL_DIR" \
  --input examples/multiple.json --mode causal --tensor-parallel-size 1 \
  --output outputs/vl-vllm-image.json
```

vLLM 使用已有的自定义 pooling／attention hooks，并非原版 OpenAI 兼容服务；容器内部命令使用 `python3`。

## 可复现的图片＋视频冒烟流程

这条流程使用 `configs/qwen3_vl_4b_hf.json` 中固定的**官方基础模型**，与上方 demo 下载的权重独立。若要显式下载该基础模型的精确版本：

```bash
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17
```

生成一张红色矩形图片和一段静态短视频，每份媒体有三道人工编写的合成题。不下载第三方媒体。**这是集成测试 fixture，分数不能用于宣传自然数据 benchmark 准确率。**

```bash
python -m mjev.benchmark.visual_smoke --root data/visual-smoke
python -m mjev.benchmark.grouped --manifest data/visual-smoke/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/vl-hf-smoke
```

运行器通过标准 Hub 缓存解析固定模型 revision，输出 `run.json`、`predictions.jsonl`、`raw.jsonl`、`report.json`、`REPORT.md` 和 `_SUCCESS`，记录源码哈希、可用的 Git 状态、依赖、媒体哈希、数值配置和计时边界。demo 的 `--local-dir` 下载目录与运行器使用的标准 Hub 缓存相互独立。

正式评测可传入冻结的[统一 manifest](benchmark.zh-CN.md)，保留原问题、候选和答案，模态为 `image` 或 `video`。同一运行器计算 Accuracy、NLL、Brier Score 和 ECE。VL 遇到包含音频的 manifest 会拒绝，而非过滤或改标注。每次运行使用新的输出目录。

在固定版本 vLLM 环境中替换为 `configs/qwen3_vl_4b_vllm.json`；启动 grouped 运行器前关闭两个 runtime hook 标志，运行器自行启用所需 hooks。

## 缓存与批处理验证

```bash
PYTHONPATH=. python benchmarks/integrated/validate_model.py \
  --manifest data/visual-smoke/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/vl-hf-cache \
  --repeats 2 --atol 0.0001
bash scripts/test.sh --suite hf -q
```

真实权重探针在 causal／isolated 两种模式下，对比逐题、批处理、缓存逐题、缓存批处理，检查缓存命中、问题顺序反转、概率和、答案一致性及最大 logits 差异。HF 每次调用包含新建 prefill；vLLM 使用已准备输入并可能保留跨调用热缓存，两者计时边界不同，记录中明确说明。

候选顺序变化可能改变模型偏好，隔离不意味着排列不变。小型原生模型测试检查：隔离时修改一个候选不会影响另一个候选的隐藏状态，但答案位置仍可响应。标签在实际官方 tokenizer 与答案边界下逐一验证为单 token。

## 验证范围

实际完成项目及限制见[最新验证入口](validation_current.zh-CN.md)。tiny 随机权重测试只验证机制，不验证预训练准确率。实验 HTTP／Tree-KV runtime 是独立路径，本次适配不会自动为其赋予 Qwen3-VL 验证结论。

官方模型：[Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)、[Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)。

### 输入与失败恢复约定

两条主后端均默认 `causal`；题内候选隔离需显式指定 `isolated`。不同问题分支独立，与同一道题的候选隔离是两种机制。候选隔离可用于可控结构比较和后续训练实验，尚不能据此断言效果上限更高。

CLI 预检查、正式推理和评分 API 共用结构校验：候选字典必须按 A/B/... 顺序排列，问题列表不能为空。分组评测逐媒体组保存到 `groups/`，同时更新 `progress.json` 和累计原始结果／预测；后续失败保留这些文件，但不生成 `_SUCCESS`。当前不支持自动断点续跑。

## mJev-Doc 文档模型

同一项目还提供 **mJev-Doc** 文档候选决策模型，共享 mJev 的 HF 核心，并提供文档输入、双语评测和 RLCD 训练入口。[中文介绍](docjev/README.zh-CN.md) · [English](docjev/README.md) · [评测结果](docjev/results.md)。mJev-Doc 权重将与 mJev 一样通过 Hugging Face 发布，两个模型的代码和说明统一维护在本仓库。当前示例支持完整本地 checkpoint。
