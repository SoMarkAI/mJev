[English](index.md) · **简体中文**

# mJev · Multimodal Decisions

一份多模态上下文，多个问题，直接输出候选答案概率。

mJev 为图片、音频和视频上的动态选择题提供统一评分接口。图片／视频从已发布的 **mJev-Qwen3-VL-4B-RLCD** 开始，需要音频时选择官方 Qwen3-Omni Thinker 权重。候选标签分数直接来自原生 LM Head。

## 适合什么场景？

- 同一图片、音频或视频需要回答多道选择题。
- 需要每个候选的原始分数、归一化概率和可复现的决策规则。
- 希望比较普通 causal、候选隔离以及前缀缓存对结果和性能的影响。

复用同一份媒体上下文，让多道题共享前缀计算，同时保留每道题的独立评分。已有记录的具体模型与配置见[最新验证入口](validation_current.zh-CN.md)。

## 从这里开始

- [中文 README 与快速开始](../README.zh-CN.md#quick-start)
- [模型选择与部署](models.zh-CN.md)
- [运行示例与已有输出记录](../examples/README.zh-CN.md)
- [HF 部署：无需 vLLM](hf.zh-CN.md)
- [vLLM Docker 部署](vllm.zh-CN.md)
- [开发者文档](development.zh-CN.md)
- [实测边界与数值稳定性](stability.zh-CN.md)
- [贡献指南](../CONTRIBUTING.zh-CN.md)

## 复现与最新验证

- [公开小型评测：数据准备 → 推理 → 报告](reproduce.zh-CN.md)
- [统一最新验证入口与后端边界](validation_current.zh-CN.md)
- [pytest 测试范围与入口](testing.zh-CN.md)
- [隔离环境安装实测与限制](installation_validation.zh-CN.md)

## 继续探索

- [已发布模型：训练与奖励设计](training.zh-CN.md)
- [缓存性能：耗时与测量条件](cache_scaling.zh-CN.md)
- [实验 HTTP / Tree-KV](integrated.zh-CN.md)
- [历史音视频工具与 Docker 镜像](../experiments/av/README.zh-CN.md)

## mjev-doc 文档模型

同一项目还提供 **mjev-doc** 文档候选决策模型，共享 mJev 的 HF 核心，并提供文档输入、双语评测和 RLCD 训练入口。[中文介绍](docjev/README.zh-CN.md) · [English](docjev/README.md) · [评测结果](docjev/results.md)。mjev-doc 需要完整本地 checkpoint；下载 mJev 权重不会得到 mjev-doc 权重。
