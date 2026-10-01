[English](av_pilot.md) · **简体中文**

# 音视频试运行（历史实验记录）

> 历史结果按当时配置记录；后续检查与缺失的配置证据见[最新验证](validation_current.zh-CN.md)。

> 这是早期 GPU 验证尚未执行时的记录。后续全权重结果见 [配对评测](paired_backends.zh-CN.md) 和 [稳定性回归](stability.zh-CN.md)。

将原始 LM Head pooling 扩展到音频和带音轨视频；不生成文本，不改变权重。保留官方 Thinker 音频/视觉编码器、processor、MRoPE 和 vLLM scheduler。attention patch 不变，候选后缀保护逻辑先检查多模态展开再移动 spans。

试验选择 10 道媒体完整的原题：MMAU 声音/音乐/语音各一题（不超过 30 秒）；最短完整 Video-MME-v2 组的四题；三个不同 MMOU 视频各一题（不超过 30 秒）。选择不查看标签，仅此 pilot 排除时长警告文件。每个完整视频采八帧、最大 65536 像素，使用整段音轨；证据时间戳和参考标签不进入推理。仅验证可行性，不代表 benchmark 总体成绩。

当时 19 项 CPU 单测通过，全部 10 题官方 processor 预检通过（154–3027 tokens）；GPU 内核、评分、隔离与缓存检查待完成，未声称音视频准确率。

## 构建和执行

从仓库根目录构建：

```bash
docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .
mkdir -p outputs/av-pilot
```

设置 `MODEL_DIR` 为本地官方模型，`BENCH_ROOT` 为已完成数据包，然后 CPU 预检：

```bash
docker run --rm -v "$MODEL_DIR:/model:ro" -v "$BENCH_ROOT:/bench:ro" \
  -v "$PWD/outputs/av-pilot:/out" mjev-av-pilot:0.1 \
  --model /model --root /bench --output /out/preflight --check-only
```

仅在四张 GPU 可用时运行固定种子的内核检查和 GPU pilot：

```bash
docker run --rm --gpus all --entrypoint python3 mjev-av-pilot:0.1 experiments/av/av_witness.py

docker run --rm --gpus all --ipc=host -e VLLM_BATCH_INVARIANT=0 \
  -v "$MODEL_DIR:/model:ro" -v "$BENCH_ROOT:/bench:ro" \
  -v "$PWD/outputs/av-pilot:/out" mjev-av-pilot:0.1 \
  --model /model --root /bench --output /out/gpu --numerics native --model-revision "$MODEL_REVISION"
```

同一准备输入比较 causal/isolated，并检查等 token 长度干预的隐藏状态隔离、causal 正对照、缓存复用、候选重排和三个并发视频问题。合成机制测试选项不纳入准确率。失败停止并保存 `failure.json`，不自动重试；实际 GPU 断言全部通过才写 `_SUCCESS`。

报告 Accuracy/NLL/Brier/ECE、原 logits/概率与延迟。延迟排除解码/准备和加载，包含 encode 请求，启用缓存并交替模式顺序；十次计时仅作描述。该 pilot 音视频上限 8192 tokens，图片仍为 4096，预检拒绝超限，不静默截断长媒体。

设置 MODEL_REVISION 为本地模型真实 commit；native 是明确选择的重跑配置，不能填补历史证据缺失。
