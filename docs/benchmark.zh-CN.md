# 通用评测

[English](benchmark.md) · [文档评测](docjev/evaluation.md) · [HF 推理](hf.zh-CN.md)

文档任务使用 `docjev.evaluate`；通用图片、视频与音频任务使用本页的冻结 manifest。数据下载与训练独立于推理仓库。

## 输入

JSONL 每行一题，媒体路径相对 manifest。保留原始问题、候选顺序与参考答案：

```json
{"id":"q1","dataset":"my-data","source":{"revision":"my-frozen-version"},"task":"color","modality":"image","media_path":"page.png","media":{"type":"image","path":"page.png"},"question":"What color is the rectangle?","candidates":{"A":"Red","B":"Blue"},"label":"A"}
```

`modality` 可为 `image`、`video`、`audio` 或 `audio_video`。Qwen3-VL 只处理图片和无音频视频；音频任务使用 Qwen3-Omni。参考标签、原始标注和证据时间戳不传给模型，也不用于裁剪媒体。

## 运行

使用已安装的 HF 环境，从仓库根目录执行：

```bash
python -m mjev.benchmark.grouped --manifest data/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/evaluation
```

配置固定官方模型 ID、40 位 revision、数值模式、投影、缓存、batch 和媒体参数。运行器下载或复用该 revision 的 Hub 缓存。vLLM 使用 `configs/qwen3_vl_4b_vllm.json` 和已安装的固定 vLLM 环境。文档微调权重通过 [docjev 评测](docjev/evaluation.md)指定本地模型。

输出目录必须是新目录。`run.json` 保存源码、模型配置和媒体哈希，`predictions.jsonl` 保存原始候选 logits，`report.json` 保存指标，`groups/` 与 `progress.json` 保留已完成题组。失败保留结果并写 `ERROR.json`，全部完成才写 `_SUCCESS`；分组运行器不支持自动续跑。

离线重算：

```bash
python -m mjev.benchmark.evaluate --manifest data/manifest.jsonl \
  --predictions outputs/evaluation/predictions.jsonl --output outputs/metrics.json
```

已有自定义评分函数可使用 `python -m mjev.benchmark.run --help`；该入口支持绑定 manifest 与配置的续跑。模型接收的输入由 `MJevDataset.model_input()` 限定。

## 指标与检查

报告 Accuracy、NLL、Brier Score 和 ECE。未知或重复 ID、标签不匹配、缺失预测默认失败；不静默重归一化概率。并列分数按输入候选顺序决定。只对实际样本范围报告成绩。

合成集成测试需 FFmpeg：

```bash
python -m mjev.benchmark.visual_smoke --root data/visual-smoke
python -m mjev.benchmark.grouped --manifest data/visual-smoke/manifest.jsonl \
  --config configs/qwen3_vl_4b_hf.json --out outputs/visual-smoke
```

这会生成自有矩形图片和静态视频；分数不是自然数据 benchmark 准确率。[缓存对照](cache_scaling.zh-CN.md)另行检查串行、batch 和前缀复用。
