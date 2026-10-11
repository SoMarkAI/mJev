# 数值模式与前缀缓存

[English](stability.md) · [HF](hf.zh-CN.md) · [vLLM](vllm.zh-CN.md) · [性能记录](cache_scaling.zh-CN.md)

## HF

通用 HF 默认 `stable`；文档模型默认 `native`、单题、无缓存。`stable` 保持加载权重的 dtype，文本计算使用 FP32，并对齐 attention 与投影的计算分块。数值结果可能与 native BF16 不同；同一对照必须固定数值模式、投影和媒体参数。

```bash
python demo_hf.py --model /path/to/model --input examples/multiple.json \
  --numerics stable --projection full --question-batch-size 2 --prefix-cache
```

每道题是独立 batch 行，共享前缀复制到各自的 KV 分支，不读取其他问题的答案。缓存复用前缀计算，也会占用显存；它不是零复制缓存。文档 HTTP 服务另有 worker 内跨请求缓存，见[服务指南](docjev/serving.zh-CN.md)。

`causal` 为默认题内注意力；`isolated` 显式限制同题候选之间的可见性。问题分支独立与题内候选隔离是不同机制。

## vLLM

进程启动前设置 `VLLM_BATCH_INVARIANT=1`。mJev hooks 配合稳定线性/MoE 内核、RMSNorm 与对齐 SDPA，保留调度和候选 mask。

```bash
VLLM_BATCH_INVARIANT=1 python demo.py --model /path/to/model \
  --input examples/multiple.json --mode causal
```

vLLM 与 HF 使用不同执行路径，不保证跨后端逐位相等。冷请求也不保证相互命中缓存。

## 验证

[开发测试](development.zh-CN.md#tests)检查随机小模型机制；[缓存检查](hf.zh-CN.md#评测与验证)比较真实权重的串行、batch 和缓存输出。报告最大 logit/概率差、答案一致性和计时边界，不只看答案是否一致。

完整历史数值回归见[固定记录](https://github.com/SoMarkAI/mJev/tree/a313f135041cf893db27a81f40052b9e66b2b5a2/docs/stability.zh-CN.md)，其模型、依赖、样本和硬件范围不代表所有部署环境。
