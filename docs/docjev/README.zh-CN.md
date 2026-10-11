# mJev-Doc · 安装与使用

[English](README.md) · [项目首页](../../README.zh-CN.md)

输入文档图片、问题与候选项，使用共享 HF 推理核心返回候选 logits、相对概率和最终选择。

[输入输出](input-output.md) · [图片服务](serving.zh-CN.md) · [Benchmark](benchmark.zh-CN.md) · [评测结果](results.zh-CN.md) · [模型背景](training.md)

<a id="installation"></a>

## 安装

Linux、Python 3.11+、NVIDIA GPU。先安装与驱动兼容的 PyTorch 2.11；HF 路径固定 Transformers 5.13.1。纯图片推理不需要 vLLM、FFmpeg 或自定义 CUDA 编译。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[docjev]' huggingface_hub
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
hf auth login
hf download SoMarkAI/mjev-doc --local-dir models/mjev-doc
python demo_docjev.py --model models/mjev-doc \
  --input examples/docjev/multiple.json --device-map cuda:0 \
  --output outputs/mjev-doc.json
```

下载目录需含完整权重分片、索引、配置、tokenizer、processor 与 chat template。推理使用本地文件；不要将 Token 写入输入文件或仓库。

## 多问题与缓存

```bash
python demo_docjev.py --model models/mjev-doc \
  --input examples/docjev/multiple.json --numerics stable \
  --prefix-cache --question-batch-size 2 --output outputs/cached.json
```

缓存／批量模式使用 `stable`，与 `native` 的数值配置不同。模型发布身份以 Hugging Face 的 revision 和模型卡为准。mJev-Doc 选用全参 SFT checkpoint；[模型背景](training.md)说明权重来源。

## 常见问题

| 问题 | 检查方式 |
|---|---|
| 找不到模型 | 检查下载目录和完整分片 |
| Transformers 版本不符 | 在干净环境安装固定版本的项目依赖 |
| CUDA 不可用 | 核对驱动、PyTorch CUDA 版本与 GPU 可见性 |
| 显存不足 | 先用单题、无缓存；必要时明确降低像素预算 |
| 输入超过 4000 tokens | 减少图片预算或任务文本，不静默截断 |
| 候选标签不是单 token | 减少候选数，重新执行预检 |
| 缓存参数被拒绝 | 增加 `--numerics stable`，与同配置无缓存结果比较 |

改变像素预算或数值模式会改变评测条件。Windows 与动态批处理见[服务指南](serving.zh-CN.md)。仅离线评分可用 `python -m docjev.evaluate --predictions outputs/evaluation.predictions.jsonl --output outputs/recomputed.json`。[开发与测试](../development.zh-CN.md#tests)另列环境与命令。
