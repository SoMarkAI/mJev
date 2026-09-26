[English](index.md) · **简体中文**

# mJev · Multimodal Decisions

一份多模态上下文，多个问题，直接输出候选答案概率。

mJev 支持官方 Qwen3-VL-4B 与 Qwen3-Omni Thinker，为图片、音频和视频上的动态选择题提供统一评分接口。直接读取原始 LM Head 的候选标签分数。

## 适合什么场景？

- 同一图片、音频或视频需要回答多道选择题。
- 需要每个候选的原始分数、归一化概率和可复现的决策规则。
- 希望比较普通 causal、候选隔离以及前缀缓存对结果和性能的影响。

从单卡已验证的 Qwen3-VL-4B 图片／视频示例开始，需要音频时切换到 Omni。复用同一份媒体上下文，让多道题共享前缀计算，同时保留每道题的独立评分。

## 从这里开始

- [中文 README 与快速开始](../README.zh-CN.md#quick-start)
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
