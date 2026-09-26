[English](benchmark.md) · **简体中文**

Docker 之外的命令均假定已激活 Python 3.11+ 虚拟环境：先用 `python3 -m venv .venv` 创建，再运行 `source .venv/bin/activate`。激活后安装和运行统一使用 `python`；Docker 内使用 `python3`。Shell 脚本也支持 `PYTHON=/path/to/venv/bin/python`。历史执行记录保留原始命令。

# 小规模音视频 benchmark

`mjev.benchmark` 构建冻结、保留原标签的评测集，不训练或重生成标注。媒体和标注留在 Git 外，数据许可与 mJev 代码许可分开。

| 来源 | 题数 | 选择单位 | 候选数量 |
| --- | ---: | --- | --- |
| [MMAU-test-mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) | 1000 | 全部 mini 样本 | 保留，含非四选项 |
| [Video-MME-v2](https://huggingface.co/datasets/MME-Benchmarks/Video-MME-v2) | 1000 | 250 视频，每视频全部四题 | 保留 |
| [MMOU Test Mini](https://huggingface.co/datasets/nvidia/MMOU) | 1000 | 固定 Test Mini 视频索引有媒体的题 | 当前来源为 10 |

按 `SHA256(seed:dataset:source_id)` 排序，seed 为 `20260925`，不依赖参考答案或网络时序，重排源行不改变抽中 ID。Video-MME-v2 保留完整题组；这里评估器报告逐题指标，不是官方题组榜单分数。不能把子集成绩称为全 benchmark 成绩。

固定 MMOU 标注有 5000 题，九个参考视频不在固定 `test/` 媒体索引。记录九个排除 ID，从剩余 4991 题抽样，默认需要 903 个不同视频。下载失败不静默替换。

## 准备与下载

工具不导入 torch/vLLM，checkout 内 Python 3.11+，安装轻依赖：

```bash
python -m pip install 'requests>=2.31' 'pyarrow>=15'
export BENCH_ROOT="$HOME/datasets/mjev_av_mini_v1"
python -m mjev.benchmark.prepare --output "$BENCH_ROOT"
python -m mjev.benchmark.fetch --root "$BENCH_ROOT" --workers 8
```

安装 FFmpeg 后，`BENCH_ROOT=/path/to/data bash scripts/build_av_benchmark.sh` 依次准备、下载、审计，失败即停。

prepare 仅下载标注/元信息；`prepare.py` 固定 revision，`selection.json` 通过 manifest 记录 ID、采样、排除与哈希。已有目录若 seed/数量不同则拒绝，新方案用新目录。

fetch 可用 `--dataset mmau|video_mme_v2|mmou` 和 `--limit N`（下载任务数，不是题数）做 pilot，严格下载冻结计划：

- MMAU：mini Parquet 的 1000 音频提取原始字节，不解码重编码。因含 WAV/MP3，用 `.audio` 中性后缀，按内容解码。含标签前缀的原始问题/选项存于 original。
- Video-MME-v2：HTTP Range 读取 ZIP 目录/header 和选中 MP4，不回退下载完整 ZIP；校验 CRC 和大小。小段 ZIP 框架读取可包含相邻字节，但不提取/保存未选视频。
- MMOU：仅从 [媒体仓库](https://huggingface.co/datasets/sonalkum/MMOU-Videos) 下载选中 `test/<video_id>.mp4`，不下载全部语料或 captions。

同一任务/文件一次只允许一个下载器负责。完成项有 SHA256 回执；直链失败保留 `.part` 供显式恢复；ZIP 中断仅重启该成员。无自动重试：错误停止新提交，等待进行中任务并记录暂停。重跑同命令为显式恢复。

可通过 `MJEV_HF_ENDPOINT` 选 HTTPS 镜像，仅改变传输，不改变记录的官方 URL、revision、ID。仓库配置不写代理、token 或私有地址。媒体时长差异大，千题可能需大量磁盘；构建器不截断帧/音频。

## 数据目录与结构

```text
$BENCH_ROOT/
├── manifest.jsonl          # 3000 records, paths relative to this directory
├── mmau.jsonl             # 1000 records
├── video_mme_v2.jsonl      # 1000 records
├── mmou.jsonl              # 1000 records
├── selection.json          # sampling, revisions, counts, annotation hashes
├── download_plan.json
├── download_status*.json
├── sources/                # source annotations and mini audio Parquet
├── media/{mmau,video_mme_v2,mmou}/
├── media_index.json        # produced by audit: hashes, streams, durations
├── validation.json
└── _SUCCESS                # only after final data audit passes
```

下例是结构示意，不是原 benchmark 题：

```json
{
  "id": "mmou:original-question-id",
  "dataset": "mmou",
  "source": {"repository": "nvidia/MMOU", "revision": "pinned-commit", "file": "MMOU_TEST_MINI.json", "original_id": "original-question-id"},
  "split": "test-mini",
  "task": ["Temporal Understanding"],
  "modality": "audio_video",
  "media_path": "media/mmou/example.mp4",
  "media": {"type": "video", "path": "media/mmou/example.mp4"},
  "question": "Original question text",
  "candidates": {"A": "Original first option", "B": "Original second option"},
  "label": "B",
  "metadata": {},
  "original": {}
}
```

candidates 为保序映射，保留顺序、文本、数量与答案。仅拆出 `A. `、`(A) ` 等显式标签分隔符，精确原形式存 original 并通过重建审计；不模糊匹配或重标注。原证据时间戳保留为元信息，**绝不自动传入推理或用于裁剪**。

## Loader 与推理接口

```python
from mjev.benchmark.dataset import MJevDataset

suite = MJevDataset("/path/to/benchmark/manifest.jsonl")
for record in suite:
    payload = suite.model_input(record)
    messages = suite.qwen_messages(record)
```

安装 `qwen-omni-utils[decord]==0.0.9`、`librosa==0.11.0`、`audioread==3.0.1` 的 Qwen 环境可使用适配器构造官方 vLLM 多模态输入并验证上下文标签：

```python
from transformers import AutoProcessor
from mjev.benchmark.adapters import qwen_vllm_input

processor = AutoProcessor.from_pretrained("/path/to/official/model", local_files_only=True)
record = suite.records[0]
prompt, label_token_ids = qwen_vllm_input(
    suite, record, processor,
    use_audio_in_video=(record["modality"] == "audio_video"),
    video_options=None,  # set and record an explicit policy for video runs
)
# Pass prompt to an AV-capable scoring backend; see av_pilot.md for the experimental mJev path.
```

遵守 [官方 audio-in-video 合约](https://github.com/QwenLM/Qwen3-Omni#use-audio-in-video)，预处理与后端 `use_audio_in_video` 必须一致。patch 大小读官方 processor，传入工具实际采样 FPS，不用默认值静默覆盖时间元信息。label_token_ids 对应精确 `Answer:\n` prefill 后下一 token。

model_input 仅暴露 id、modality、解析后的 media_path、question、candidates，剔除标签、原标注、任务和证据位置。qwen_messages 为官方 processor 提供媒体内容及原题选项；后端负责解码。音视频使用官方 audio-in-video，明确处理无声视频，记录帧采样/缩放/音频截断策略。loader 不附加字幕或裁判 captions。

实现接收 payload、返回标签到分数映射的 scorer：

```python
async def score(payload):
    # Run your AV-capable backend here; do not read labels from the manifest.
    result = await backend.score(payload)
    return {"logits": result.raw_candidate_logits}
```

可恢复运行：

```bash
python -m mjev.benchmark.run \
  --manifest "$BENCH_ROOT/manifest.jsonl" \
  --scorer your_backend:score --run-config configs/your_av_run.json \
  --concurrency 3 --output outputs/av_predictions.jsonl
```

自备 run config JSON，标明模型 revision、模式、精度和媒体预处理（帧/fps/像素、音频、任何截断）。`.run.json` 将预测绑定配置、scorer 与 manifest 哈希；设置变化拒绝续跑。运行器只记录配置，后端必须实际执行。每模型/配置使用新输出路径。

每行输出必须含 id 和 logits 或 probabilities，覆盖全部原标签。`adapters.from_mjev_result` 校验选项文本/顺序后转换；`adapters.from_qwen_logits` 从原始下一 token 词表向量选标签，并在精确模板/prefill 验证单 token。均不 generate 或改变权重。

**历史边界：**最初 mJev 0.1 的 MJevEngine 仅支持图片，当时此工具包只建立音视频数据与接口，不认证 GPU 音视频隔离。后续已扩展，当前能力见 [HF 文档](hf.zh-CN.md) 与 [配对评测](paired_backends.zh-CN.md)；数据/工具测试本身仍不代表模型准确率。

## 评估

```bash
python -m mjev.benchmark.evaluate \
  --manifest "$BENCH_ROOT/manifest.jsonl" \
  --predictions outputs/av_predictions.jsonl \
  --bins 15 --output outputs/av_metrics.json
```

- Accuracy：argmax 正确率，精确并列按原候选顺序。
- NLL：参考概率的负自然对数均值；logits 用稳定 log-sum-exp。真零目标概率产生无穷，序列化为 null 且 `nll_is_infinite=true`，不偷偷裁剪。
- Brier：每题候选概率与 one-hot 真值平方差之和，再取平均，不除候选数，范围 0–2。
- ECE：最高标签置信度，默认 15 个等宽桶，左闭右开、末桶含 1，受分桶和样本量影响。

概率须有限、位于 [0,1]，和在 1e-6 容差内为一，不静默重归一化。只有硬答案不足以算 NLL/Brier/ECE。未知/重复 ID、标签不匹配和缺失预测默认失败。`--allow-partial` 明确报告覆盖率与缺失 ID，仅对已评分子集算指标。按数据集、模态、任务、候选数细分；多任务题可进入多个组，组计数不可当互斥相加。

## 审计与测试

安装 FFmpeg 中的 ffprobe 后执行：

```bash
python -m mjev.benchmark.audit --root "$BENCH_ROOT"
python -m pytest tests/test_benchmark.py -q
```

审计逐题重建原题/选项/答案，检查冻结 manifest 哈希、文件大小/SHA256 回执，用 ffprobe 检查流和时长，发现缺失/额外媒体并记录无音轨视频。`_SUCCESS` 只说明数据审计通过，不是推理完成。仅容器/流检查，不是每个音频样本/视频帧的完整解码。

原数据包统计与来源限制见 [数据验证](benchmark_validation.zh-CN.md)。时长差异另存 `duration_warnings.json`，不覆盖原元信息。
