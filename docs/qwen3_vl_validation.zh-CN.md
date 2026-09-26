[English](qwen3_vl_validation.md) · **简体中文**

# Qwen3-VL-4B 验证记录 — 2026-09-26

两条主后端均已各用单张 24 GB NVIDIA GPU 跑通真实 **Qwen3-VL-4B-Instruct** 图片／视频推理。这是小规模集成与一致性验证，不证明总体 benchmark 优势。[安装与复现](models.zh-CN.md)。

## 环境与来源

- 官方 revision：`ebb281ec70b05090aa6165b016eac8ec08e71b17`。两个权重分片逐个核对 SHA-256，与固定 Hub 元数据一致：[校验记录](validation_evidence/qwen3-vl-weight-integrity.json)。
- HF：全新 Linux Python 3.11.16 venv，不继承系统 site-packages；安装 `.[hf-vl,test]` 和官方 TorchCodec 0.11.0+cpu。复用 pip 下载缓存，未安装 vLLM 或 `qwen_omni_utils`。依赖检查、固定随机种子的 CUDA 运算、123 项 CPU 测试通过，3 个可选 vLLM 模块跳过。
- vLLM：复用固定 0.25.1／Transformers 5.13.1 的现有运行环境，Python 3.12，130 项 CPU 测试通过。**没有重新构建干净 vLLM 镜像**。解码器版本与 HF 不同，各报告记录完整依赖。
- 新的独立 reviewer 在新装环境中执行文档的 HF 预检与缓存 demo，均通过；reviewer 没有重复安装或下载权重。
- 代码基线 `2e75abf` 加本次适配改动。报告记录实际 dirty-tree 源码哈希和可用的 Git 信息；vLLM 精简镜像没有 Git 可执行文件，相关字段显式为 null，源码哈希仍保留。后续文档和 worker 启动修复不能倒算为早期 HF 运行时已包含的改动。

[机器可读汇总](validation_evidence/qwen3-vl-validation-summary.json)。

## 实际通过项目

| 检查 | HF | vLLM |
| --- | --- | --- |
| 真实权重图片＋视频合成冒烟 | 6/6 题处理完成 | 6/6 题处理完成 |
| causal 与候选隔离模式 | 通过 | 通过 |
| 逐题／批处理／缓存逐题／缓存批处理 | 通过 | 通过 |
| 缓存 token 复用、概率归一化 | 通过 | 通过 |
| 问题顺序反转 | 通过 | 通过 |
| 候选顺序反转映射 | 通过，不要求答案不变 | 通过，不要求答案不变 |
| 本次执行方式对比的最大 logits 差值 | **0.0** | **0.0** |
| NExT-QA 原始答案视频小样本 | 18/18 题处理完成 | 18/18 题处理完成 |

后端内缓存对比使用 stable、完整 LM Head 投影及两种 attention 模式，每种计时配置在一次预热后执行两遍。结果不保证任意输入与 batch 大小都相同。HF 将前缀 KV 复制到独立分支；vLLM 也可能缓存重复问题及候选 token，因此热请求计时**不能单独证明公共媒体前缀带来的加速**。

小型原生模型另测候选隐藏状态隔离、答案位置可见性、异常后的 hook 清理和权重不变，覆盖 FP32／BF16。VL 拒绝音频，Omni 音频路径保留。

证据：[HF 缓存](validation_evidence/qwen3-vl-vl-hf-cache-report.json)、[vLLM 缓存](validation_evidence/qwen3-vl-vl-vllm-cache-report.json)、[HF 合成冒烟](validation_evidence/qwen3-vl-vl-hf-smoke-report.json)、[vLLM 合成冒烟](validation_evidence/qwen3-vl-vl-vllm-smoke-report.json)。

## 原始答案视频小样本

从已有 200 视频子集中，按 video ID 字典序选取前两个视频，保留全部 10＋8 道 NExT-QA validation 原题及答案，不按标签筛选。这是已有数据的小样本验证；仓库不分发媒体／标注，也不声称已经提供此选择的公开自动下载复现路径。可移植、无需第三方数据的入口是单独的合成冒烟流程。

| 指标 | HF | vLLM |
| --- | ---: | ---: |
| 正确／总数 | 8 / 18 | 7 / 18 |
| Accuracy | 44.44% | 38.89% |
| NLL | 1.323149 | 1.298714 |
| 多分类 Brier | 0.698866 | 0.695358 |
| ECE，15 bins | 0.306792 | 0.326210 |
| 两组耗时，不含加载 | 2.89 秒 | 15.04 秒 |
| 模型启动 | 3.92 秒 | 33.40 秒 |

这是**未预热的小样本冷启动计时**，包含解码、预处理、评分；vLLM 首个请求还遇到运行时启动工作，不能据此给后端整体速度排名。两后端保留各自原生处理与数值路径：此视频样本中 HF prompt 长度多两个 token，所以跨后端 logits、答案及准确率**不属于等价性验证**。这些样本不能证明 HF 普遍更准，也不能证明 4B 优于 Omni。

证据：[HF 视频报告](validation_evidence/qwen3-vl-vl-hf-nextqa-report.json)、[vLLM 视频报告](validation_evidence/qwen3-vl-vl-vllm-nextqa-report.json)。

## Omni 回归与保留的失败

重新运行原 Omni HF 公开六题，每个候选 logits 与原报告完全相同，最大绝对差 **0.0**，原始答案准确率仍为 **2/6**。验证的是共享适配层在该任务上的回归，不代表新增 Omni vLLM 验证。[回归报告](validation_evidence/qwen3-vl-omni-regression-report.json)。

首次 VL vLLM 运行因 worker 加载容器旧目录、没有注册 `MJevVL` 而失败。现在引擎启动会向 worker 传入当前源码目录及 opt-in hooks。失败目录与成功目录分开保存，失败不并入分数。原始日志和第三方源标注保留在 Git 外。

实验 HTTP／Tree-KV、VL 音频、其他模型大小、量化和广泛准确率／性能结论不在本次验证范围内。

最终缓存探针在每次预热后计时重复两次，逐次检查并保存数值比较，同时保存问题倒序的比较结果；两后端各 32 次重复比较均为最大 logits 差值 0。
