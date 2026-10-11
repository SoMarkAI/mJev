# Hugging Face 推理

[English](hf.md) · [文档模型](docjev/README.zh-CN.md) · [服务](docjev/serving.zh-CN.md)

Linux、Python 3.11+、NVIDIA GPU。固定 PyTorch 2.11.0、Transformers 5.13.1；直接使用模型 forward 返回候选 logits。

<a id="models"></a>

## 模型与安装

- `SoMarkAI/mjev-doc`：文档图片，使用[文档指南](docjev/README.zh-CN.md)。
- `SoMarkAI/mJev-Qwen3-VL-4B-RLCD`：通用图片和视频，基础架构为 Qwen3-VL。
- `Qwen/Qwen3-Omni-30B-A3B-Instruct`：另支持音频与带音轨视频，需要更大显存；历史配置为四张 24 GB GPU。

在虚拟环境中安装，视频需系统 FFmpeg：

```bash
python -m pip install torchcodec==0.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[hf-vl]' huggingface_hub
hf download SoMarkAI/mJev-Qwen3-VL-4B-RLCD --local-dir models/mJev
```

Omni 使用 `python -m pip install -e '.[hf]'`，并下载完整 Omni checkpoint。保留配置、tokenizer、processor、chat template 与全部权重分片。

## 图片与视频

```bash
python demo_hf.py --model models/mJev --input examples/multiple.json \
  --mode causal --numerics stable --projection full \
  --prefix-cache --question-batch-size 2 --output outputs/image-decisions.json
python demo_hf.py --model models/mJev --input examples/motion-demo/input.json \
  --numerics stable --projection full --output outputs/video-decisions.json
```

输入路径相对 JSON 文件，格式见[示例](../examples/README.zh-CN.md)。`--check-only` 加载 processor/tokenizer、解码媒体并检查实际输入，不加载模型权重；默认输入上限 4000 tokens，超限报错。

音视频输入使用 `modality: "audio"`、`"video"` 或 `"audio_video"`，文件路径放在 `media_path` 字段。`video_options` 可明确 `fps`、`min_pixels`、`max_pixels` 和 `max_frames`。Qwen3-VL 拒绝音频任务，不静默丢弃音轨；`audio_video` 需 Omni。

## 评分与缓存

默认 `causal`；`isolated` 显式启用题内候选隔离。候选必须在实际模板边界对应单 token，返回原始 logits、候选 softmax 和确定性并列处理结果。

`stable` 是通用 HF 默认数值模式。批处理和前缀缓存比较必须固定数值模式与投影，详见[数值与缓存](stability.zh-CN.md)。`--projection full` 使用完整词表投影；默认 `selected` 只计算候选行。`--device-map`、`--dtype` 和 `--max-input-tokens` 可显式配置。

## 评测与验证

[通用评测](benchmark.zh-CN.md)保留原始标签与候选顺序；[文档评测](docjev/evaluation.md)支持本地 checkpoint 和候选重排。

```bash
PYTHONPATH=. python benchmarks/integrated/validate_model.py \
  --manifest data/visual-smoke/manifest.jsonl --config configs/qwen3_vl_4b_hf.json \
  --out outputs/cache-check
```

先按[通用评测](benchmark.zh-CN.md)生成 smoke 输入。此检查使用真实权重比较串行、batch、缓存结果并记录耗时；合成题不代表 benchmark 准确率。CPU 机制测试见[开发指南](development.zh-CN.md)。
