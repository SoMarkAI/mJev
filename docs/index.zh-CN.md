[English](index.md) &nbsp;&nbsp; **简体中文**

# mJev 文档

mJev 从图片、文档和视频中为候选项评分；mJev-Doc 面向文档分类、字段核验和规则判断。两个模型共用推理接口，权重单独从 Hugging Face 下载。

## 使用

| 你想做什么 | 从这里开始 |
| --- | --- |
| 运行文档模型 | [mJev-Doc 指南](docjev/README.zh-CN.md) |
| 了解输入和输出 | [文档 API](docjev/input-output.md)、[示例](../examples/README.zh-CN.md) |
| 同时处理多张图片和多个问题 | [HF 服务、动态 batch 与前缀缓存](docjev/serving.zh-CN.md) |
| 处理通用图片、视频或音频 | [模型选择](models.zh-CN.md)、[HF 部署](hf.zh-CN.md) |
| 使用 vLLM | [vLLM 部署](vllm.zh-CN.md) |

## 评测与训练

- [DocJev-bench：任务和业务场景](docjev/benchmark.zh-CN.md)
- [文档评测、候选重排与模型对比](docjev/evaluation.md)
- [mJev-Doc 训练](docjev/training.md)、[通用 mJev 训练](training.zh-CN.md)
- [公开小型评测流程](reproduce.zh-CN.md)
- [结果与验证范围](validation_current.zh-CN.md)、[缓存性能](cache_scaling.zh-CN.md)

## 开发

- [代码入口与实现约束](development.zh-CN.md)
- [测试命令](testing.zh-CN.md)、[数值稳定性](stability.zh-CN.md)
- [参与贡献](../CONTRIBUTING.zh-CN.md)
- [实验性 vLLM HTTP / Tree-KV](integrated.zh-CN.md)、[音视频实验工具](../experiments/av/README.zh-CN.md)
