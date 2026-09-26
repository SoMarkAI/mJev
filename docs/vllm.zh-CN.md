[English](vllm.md) · **简体中文**

本页四卡配置描述 Omni；Qwen3-VL-4B 默认 TP=1，可通过 `--tensor-parallel-size` 显式指定。见[模型选择](models.zh-CN.md)。

# vLLM 部署与验证

[返回 mJev 首页](../README.zh-CN.md)

## 环境与安装

参考环境为 Linux x86-64、NVIDIA CUDA、**4 × 24 GB NVIDIA GPU**、BF16 TP4。当前配置固定为已验证 TP 大小，其他硬件尚未验证。模型约 70.5 GB，额外预留 Docker 镜像与下载缓存空间。

需要 Docker 和 NVIDIA Container Toolkit。镜像固定 vLLM **0.25.1**、PyTorch **2.11.0+cu130**、Transformers **5.13.1**，hooks 会拒绝其他 vLLM 版本。普通 stock vLLM 不支持本项目自定义候选 mask。

仓库根目录执行：

```bash
# Install the download client in your own Python environment.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
export MODEL_DIR="$HOME/models/Qwen3-Omni-30B-A3B-Instruct"
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct --local-dir "$MODEL_DIR"
docker build -t mjev:0.1.0 .
mkdir -p outputs
```

Dockerfile 固定上游镜像 digest，使用镜像已有依赖，以 `--no-deps` 安装包。同环境开发可用 `python3 -m pip install --no-deps --no-build-isolation .`，直接 `python3 demo.py` 执行下面参数。必须从 checkout 启动，worker 依赖源码 `sitecustomize.py`；不支持仅安装包后从外部目录启动。

## 运行参考后端

先检查输入、官方 processor/template 和标签 token，不向 GPU 加载权重：

```bash
docker run --rm \
  -v "$MODEL_DIR:/model:ro" -v "$PWD/outputs:/app/outputs" \
  mjev:0.1.0 --model /model --input examples/multiple.json --check-only
```

这是预检，不是推理。四张 GPU 可用后运行真实评分：

```bash
docker run --rm --gpus all --ipc=host -e VLLM_BATCH_INVARIANT=1 \
  -v "$MODEL_DIR:/model:ro" -v "$PWD/outputs:/app/outputs" \
  mjev:0.1.0 --model /model --input examples/single.json \
  --mode isolated --output outputs/isolated.json
```

推荐保持 `VLLM_BATCH_INVARIANT=1`，batch/cache 对照也应一致，见 [数值模式](stability.zh-CN.md)。`--mode causal` 为普通可见性对照，`--input examples/multiple.json` 对同图并发提问。公共前缀块可跨请求复用，但同时到达的冷请求不保证彼此命中。候选相关缓存块包含 mask 元数据，避免不兼容 mask 复用；输出 `num_cached_tokens`。

自有输入目录只读挂载到 `/inputs`，传 `--input /inputs/task.json`；图片路径相对 JSON：

```json
{
  "image": "rectangle.png",
  "context": "Inspect the supplied picture.",
  "question": "What color is the rectangle?",
  "candidates": ["Red", "Blue", "Green"]
}
```

多问题用 `questions` 数组替代顶层 `question`/`candidates`，每题保留这两个字段，共享图片与上下文，见 `examples/multiple.json`。

结果为逐题 JSON 数组，含 question、mode、candidates、decision、probability_sum、num_cached_tokens、prompt_tokens 和 spans。下例是假设数字的格式示意，不是模型结果：

```json
{
  "candidates": [
    {"label": "A", "text": "Red", "token_id": 32, "raw_logit": 2.0, "probability": 0.7310586},
    {"label": "B", "text": "Blue", "token_id": 33, "raw_logit": 1.0, "probability": 0.2689414}
  ],
  "decision": {
    "label": "A", "text": "Red", "token_id": 32,
    "raw_logit": 2.0, "probability": 0.7310586,
    "index": 0, "tied_labels": ["A"], "tie_policy": "first_in_input_order"
  },
  "probability_sum": 1.0
}
```

候选数 2–128，受真实单 token 标签校验和上下文限制。实测推理到 30 个，tokenizer 检查到 52 个；128 只是接口上限，不保证标签全通过。拒绝重复/空选项、聊天控制 token 和多 token 标签。

## 测试与验证状态

镜像内 CPU mask/decision 测试：

```bash
docker run --rm --entrypoint bash mjev:0.1.0 scripts/test.sh
```

真实模型测试需要四张空闲 GPU，初始化可能远慢于一次请求：

```bash
docker run --rm --gpus all --ipc=host --entrypoint bash \
  -e MJEV_MODEL=/model \
  -v "$MODEL_DIR:/model:ro" -v "$PWD/outputs:/app/outputs" \
  mjev:0.1.0 scripts/test_gpu.sh
```

GPU 套件检查隐藏状态干预隔离、causal 正对照、stock Triton 对比、候选重排、并发问题、概率和及 mask 感知缓存。全部断言通过才写 `outputs/e2e/runs/<run>/_SUCCESS`。候选重排不一定保持预测，位置编码和标签效应仍存在。

发布前原型在参考环境通过该套件；打包独立验证，见 [历史发布记录](validation.zh-CN.md)。仅 CPU/预检不称为新 GPU 结果，后续回归见 [稳定性文档](stability.zh-CN.md)。

## 范围与限制

- 仅 Thinker，不实例化 Talker。HF/pooling 支持图片、音频、视频、带音轨视频。
- 保留官方多模态 processor、chat template、视觉编码器、位置编码、scheduler 和 paged KV cache。
- 自定义 attention 将缓存 K/V 收集为 dense SDPA 张量，是正确性原型，不是优化内核或生产服务栈。
- eager 执行、无 chunked prefill/异步调度、KV 未量化，最大序列 4096、最多八个调度序列、图片最大 262144 像素，配置见 `mjev/engine.py`。
- 该分辨率可能丢失文档小字；不提供训练代码或权重。HTTP runtime 为独立实验实现。
