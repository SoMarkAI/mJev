[English](models.md) · **简体中文**

[真实模型验证记录](qwen3_vl_validation.zh-CN.md)

# 两个模型，同一套决策接口

mJev 根据本地官方 `config.json` 自动识别模型家族，保留各模型原生 processor、chat template、视觉特征、位置编码与 LM Head。不训练、不加 LoRA、不新增决策头、不调用 `generate()`。

| 能力 | Qwen3-Omni-30B-A3B-Instruct | Qwen3-VL-4B-Instruct |
| --- | --- | --- |
| 图片＋文本 | 支持 | 支持 |
| 视频＋文本 | 支持 | 支持 |
| 音频／视频音轨 | 支持 | **不支持，显式拒绝** |
| 动态候选、logits、概率与决策 | 统一接口 | 统一接口 |
| HF 因果／候选隔离 attention | 支持 | 支持 |
| HF 公共前缀 KV 与真实问题批处理 | 支持 | 支持 |
| vLLM pooling、mask、prefix cache | 可选后端 | 可选后端 |
| 默认 vLLM 张量并行度 | 4 卡 | 1 卡 |
| 权重 | 单独下载官方权重 | 单独下载官方权重 |

Qwen3-VL 是稠密视觉语言模型，参数较少不等于已经证明某个加速倍数。它不能替代 Omni 处理音频任务；尤其不能去掉音频相关 benchmark 的音轨后，将成绩当作同等条件的评测。

## 安装与运行 4B

文档中的 GPU／视频流程需要 Linux、Python 3.11+、NVIDIA GPU 和系统 FFmpeg。HF 无需安装 vLLM 或编译自定义 CUDA 内核；CPU codec 负责视频解码，CUDA PyTorch 负责模型推理。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl,test]' huggingface_hub
export MODEL_DIR="$HOME/models/Qwen3-VL-4B-Instruct"
hf download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 --local-dir "$MODEL_DIR"
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json --check-only
python demo_hf.py --model "$MODEL_DIR" --input examples/multiple.json \
  --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 3 --output outputs/vl-image.json
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
