[English](reproduce.md) · **简体中文**

# 公开小型评测：准备 → 推理 → 报告

这是新增的 **MMOU 短视频子集**，不是历史 1,249 题内部配对数据集的复现，也不是官方榜单成绩。不需要内部数据包、Infinity 文件或服务器路径。默认选两个视频，保留其全部原题（当前六题）；不改变选项或答案。按固定 seed 和标注时长筛选，不看参考标签。时长短不保证 token 不超限；超长或解码错误明确停止。

## 安装与固定配置

使用 [Linux HF 安装](hf.zh-CN.md)、FFmpeg 及 `pip install -e '.[hf,test]'`。vLLM 另需固定的可选依赖与 [部署环境](vllm.zh-CN.md)。所有命令从 checkout 执行。请先查看 [本次实际安装验证](installation_validation.zh-CN.md)，不要把步骤说明等同于所有平台实测通过。

`configs/public_mini_hf.json` 和 `configs/public_mini_vllm.json` 明确记录 numerics、模式、BF16 权重、投影、缓存、问题 batch、完整输入上限、seed、上下文与视频处理。新流程固定官方模型 commit `26291f793822fb6be9555850f06dfe95f2d7e695`，由本次公开 API 查询得到，**不说明历史试验曾使用该版本**。infer 用这个精确 revision 调用 snapshot_download，可复用 Hub 缓存；可能需要下载约 70.5 GB 权重。不接受浮动分支/tag，也不静默使用其他本地模型目录。

初始配置为 causal、stable、full projection、逐题、无前缀缓存。HF 可复制配置后明确开启缓存/batch；vLLM mini 暂拒绝这两项，保持串行基线清楚，不影响原后端已有能力。

先按 HF 教程安装 CPU 解码包，再安装 HF extra。`torchcodec==0.11.0+cpu` 来自官方 PyTorch CPU 索引，不是默认 PyPI；随后 HF 安装使用默认 PyPI 的 CUDA PyTorch。

## 1. 只下载选中的公开媒体

```bash
python -m mjev.benchmark.public_mini prepare \
  --root data/public-mmou-mini --videos 2 --max-seconds 15 --seed 20260926
```

必须使用新目录。下载固定 MMOU Test Mini 标注及媒体索引，对合格 video_id 按 SHA256 排序，保留选中视频所有原题，仅下载对应文件。保存排除项、来源 revision、标注/manifest 哈希及媒体回执。`_DATA_READY` 仅说明字节校验通过，不代表完整解码或推理。失败写 ERROR.json，不静默换样本、不自动重试；检查失败后用新目录显式重试。

标注保留上游 Apache-2.0 声明，原视频/音轨权利仍未核实；不入 Git、不重新授权，见 [第三方说明](../THIRD_PARTY_LICENSES.zh-CN.md)。

## 2. 官方模型推理

HF：

```bash
FORCE_QWENVL_VIDEO_READER=torchcodec MJEV_ENABLE=0 MJEV_ENABLE_PATCHES=0 \
python -m mjev.benchmark.public_mini infer \
  --root data/public-mmou-mini --config configs/public_mini_hf.json \
  --out outputs/public-mmou-hf
```

可选 vLLM pooling，须在其支持的四卡环境：

```bash
MJEV_ENABLE=0 MJEV_ENABLE_PATCHES=0 \
python -m mjev.benchmark.public_mini infer \
  --root data/public-mmou-mini --config configs/public_mini_vllm.json \
  --out outputs/public-mmou-vllm
```

运行器仅为 vLLM 后端在 engine 构造前启用 hooks。每次使用新输出目录。参考答案/证据时间戳不进入评分。失败保留已完成预测和 ERROR.json，不给部分结果生成成功报告，不支持自动续推理或静默漏题。

run.json 保存代码 commit、**当前源码逐文件哈希**、脏工作区状态、模型 revision/解析状态、processor/config 哈希、全部已安装依赖版本、Python/平台、GPU 名称、engine 参数和推理配置。源码哈希区分未提交修改与基础 commit。Hub snapshot 解析不等于逐权重分片独立校验；旧本地数据包 runner 的声明 revision 标为 declared_local_unverified。

计时按媒体组，包含解码、processor、评分，模型加载另记，没有预热，不是每题 API 延迟或稳态吞吐。HF 放置与 vLLM TP4 不同，不能称同内核速度对照。

## 3. 生成指标和报告

```bash
python -m mjev.benchmark.public_mini report --out outputs/public-mmou-hf
# 可选后端运行后：
python -m mjev.benchmark.public_mini report --out outputs/public-mmou-vllm
```

完整覆盖与 manifest 校验通过才生成 report.json、REPORT.md、_SUCCESS。共用评估器算 Accuracy/NLL/Brier/ECE。raw.jsonl 保留原 logits 和分组耗时；predictions.jsonl 保存标签到 logit 映射。共享结果时保留完整输出与采样元信息，但须遵守源许可，不发布私人路径或受限媒体。

## 历史实验

保留依赖原外部数据包的 benchmarks/integrated runner。新运行必须明确 --numerics 和 40 字符 --model-revision；evaluate_hf.py 还必须明确 projection 和 question batch size，并写运行证据。vLLM stable 启动设置 VLLM_BATCH_INVARIANT=1，native 设置 0。缺少配置证据的历史产物标为**未核实**，不套用今天默认值。历史成绩不改，见 [最新验证入口](validation_current.zh-CN.md)。
