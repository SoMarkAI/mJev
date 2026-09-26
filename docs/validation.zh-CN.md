[English](validation.md) · **简体中文**

# 0.1.0 验证记录

> 最新验证状态与后端边界统一见 [验证入口](validation_current.zh-CN.md)。本页历史结果保留原记录；未记录的模型 revision、完整依赖或运行配置标为未核实，不从当前默认值推断。

> 本页为历史发布记录，不代表当前 GPU 占用或最新验证状态。后续结果见 [配对评测](paired_backends.zh-CN.md) 与 [稳定性回归](stability.zh-CN.md)。

## 发布打包：2026-09-25

在 Dockerfile 精确固定的基础镜像上，用全新、禁网、无 GPU 容器验证：

- `pip install --no-deps --no-build-isolation .` 构建并安装 wheel。
- 三项 CPU 测试：包含最终答案的 mask 可见性、不兼容 batch 拒绝、决策并列/非有限数处理。
- 本地官方 processor/tokenizer 对单题、多题示例执行 `demo.py --check-only`。
- Docker 镜像构建及实际默认入口。
- Python 语法编译与仅源代码发布扫描。

这些不构成 GPU 推理。当时参考 GPU 被其他服务占用，未停止该服务，也未重跑打包后的 GPU demo。下面模型/attention 实现在打包前测试过；发布新增输入校验、CLI、决策元信息及可移植路径。当时仍建议生产使用前补做发布 GPU 回归。

## 早期真实模型原型

已保存的成功报告使用 vLLM 0.25.1、PyTorch 2.11.0+cu130、Transformers 5.13.1、官方 BF16 模型及四张 24 GB NVIDIA GPU。该次发布的 attention patch 和模型适配器不变。

| 检查 | 观测结果 |
| --- | --- |
| 隔离候选干预：其他候选隐藏状态 | 最大绝对变化 0.0 |
| 相同干预的 causal 正对照 | 最大绝对变化 16.375 |
| SDPA causal 与 stock Triton logits | 最大绝对差 0.3125 |
| 冷/热缓存 logits | 最大绝对差 0.375 |
| 前缀缓存复用 | 冷 0，热 256 tokens |
| 跨 mask 首个 isolated 请求 | 缓存 256 tokens，公共前缀结束于 265 |
| 重复 isolated 请求 | 缓存 368 tokens |
| 三个并发问题 | 每题复用 208 tokens |
| 真实推理候选数 | 2、3、5、7、30 |

BF16 缓存/后端比较使用 raw-logit 容差 0.5，不是严格相等。候选顺序测试检查合法映射，不证明排列不变性或准确率改善。可移植测试入口为 `scripts/test_gpu.sh`，输出被 Git 忽略。

不发布原始内部日志、内部路径、模型文件与私人评测数据。以上是历史聚合结果，不是重新复现的发布结果。
