[English](validation_current.md) &nbsp;&nbsp; **简体中文**

# 结果与验证范围

## 模型评测

| 评测 | 原始 Qwen | 训练后 | 详情 |
| --- | ---: | ---: | --- |
| mJev-Doc：3,024 条中英文记录 | 82.04% | 84.36% | [协议与结果](docjev/results.md)、[分项统计](validation_evidence/mjev-doc-unified1512.json) |
| 通用 mJev：195 道题 | 77.95% | 80.00% | [训练与评测](training.zh-CN.md) |

两行使用不同评测集。文档评测含 1,512 对中英文题目，与训练时的 500 条验证记录分开统计；使用相同处理后输入、causal/native BF16、完整 LM Head 和温度 1。中英文是配对观察，不是独立样本。

## 推理与性能

| 路径 | 已有检查 | 范围 |
| --- | --- | --- |
| HF 多图片服务 | 真实小型 Qwen3-VL forward、HTTP/spawn、动态 batch、混合长度前缀缓存 | CPU 机制检查；完整 4B 服务吞吐和 Windows CUDA 尚未实测。[服务指南](docjev/serving.zh-CN.md) |
| HF 主接口与 vLLM pooling | Qwen3-VL-4B 图片／视频、缓存、批处理及 Omni HF 回归 | 仅适用于报告中的模型与配置。[验证记录](qwen3_vl_validation.zh-CN.md) |
| HF 缓存性能 | 63／2,266-token 前缀，1／3／8／16 题，每组重复 5 次 | 长前缀 16 题为 32.62→6.29 秒；单题变慢，峰值显存增加。[完整测量](cache_scaling.zh-CN.md) |
| 实验性 vLLM HTTP / Tree-KV | 独立实现与测试入口 | 主接口结果不能替代该路径的全模型验证。[实验说明](integrated.zh-CN.md) |

`stable` 与 `native` 是不同数值配置。跨配置、后端或缓存策略比较时，应固定输入和计时范围。历史 native 缓存验收失败记录保留在 [HF 指南](hf.zh-CN.md)，不能用小模型测试代替全权重验收。

## 历史记录

以下报告保留当时的环境与限制，不代表当前版本重新执行过 GPU 评测。

- [配对后端评测](paired_backends.zh-CN.md)、[数值稳定性](stability.zh-CN.md)：部分历史模型 revision 和完整环境锁未核实，不能追认为当前默认配置。
- [Linux 安装与官方模型验证](clean_linux_validation.zh-CN.md)、[早期 macOS 安装检查](installation_validation.zh-CN.md)
- [软件回归与修复记录](reliability_review.zh-CN.md)
- [原始数据完整性检查](benchmark_validation.zh-CN.md)、[音视频 pilot](av_pilot.zh-CN.md)、[首次发布验证](validation.zh-CN.md)

重新运行可从[测试命令](testing.zh-CN.md)或[公开评测流程](reproduce.zh-CN.md)开始。
