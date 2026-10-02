[English](validation_current.md) · **简体中文**

# 最新验证统一入口

最新受控性能记录：[HF 前缀缓存规模实验](cache_scaling.zh-CN.md)。一个公开动画、63／2,266-token 前缀、1／3／8／16 题，每组重复 5 次，包含首次 prefill、显存和同输入数值一致性检查；不是准确率评测。

最新软件修复与回归检查见 [可靠性修复记录](reliability_review.zh-CN.md)。默认注意力已统一为 `causal`；原始 GPU 验证仍保留各自配置。

本页列出已完成的检查及适用范围。最新模型扩展见 [Qwen3-VL-4B HF/vLLM 验证](qwen3_vl_validation.zh-CN.md)，包含缓存／批处理和 Omni HF 回归。此前的[干净 Linux HF 验证](clean_linux_validation.zh-CN.md)已跑通公开六题全流程。代码基线为 `04cc2e0a8fd723d678f44b0c69c2e74b4fe94ea4` 加本次安装／来源记录修复，实际源码哈希已记录，不声称未修改的原 commit 已通过。

## 实现边界

| 路径 | 执行方式 | 数值与缓存范围 | 证据 |
| --- | --- | --- | --- |
| HF 主路径 | 官方 Omni Thinker／Qwen3-VL forward，不依赖 vLLM、不 generate | 显式 stable/native，逐问题复制前缀 KV，真实行 batch | [HF](hf.zh-CN.md)、[数值回归](stability.zh-CN.md) |
| vLLM pooling 主路径 | AsyncLLM.encode、原 LM Head、自定义 hooks | 固定 vLLM、Omni TP4／VL TP1、mask 感知 APC、dense SDPA | [部署](vllm.zh-CN.md)、[数值回归](stability.zh-CN.md) |
| 实验 HTTP / Tree-KV | 单步 generate 传分数 | 独立 scheduler/block table 路径，全模型 Tree-KV/dense 一致性待验证 | [实验 runtime](integrated.zh-CN.md) |
| 历史 AV pilot / 发布打包 | 仅记录对应阶段的检查 | “GPU 待验证”指当时阶段，不是当前 HF/pooling 能力 | [AV pilot](av_pilot.zh-CN.md)、[历史打包](validation.zh-CN.md) |

主路径验证不能套用到 HTTP/Tree-KV。本次修复未改模型权重、attention 计算公式或候选评分行为。

## 此前 Omni Linux 验证：2026-09-26

干净 Python 3.11 安装、依赖检查、四卡 CUDA 运算、105 项 CPU 测试（3 个可选跳过）、原 H.264／AV1 解码、6/6 官方模型 HF 推理及报告生成均通过。实际完整输入为 920–1,255 token。图片 demo processor 预检通过，未执行图片完整模型推理或新的 vLLM／cache 回归。见[完整结果与失败记录](clean_linux_validation.zh-CN.md)、[机器可读证据](validation_evidence/linux-clean-install.json)。

## 早期 macOS 修复检查：2026-09-26

以下表格保留早期特定主机的结果，不代表最新 Linux 状态。

| 检查 | 实际结果 |
| --- | --- |
| 全新 Python 3.11 macOS ARM64 环境按 HF extra 安装 | **依赖解析失败**：此平台无 decord 分发包 |
| 同一隔离环境安装 `pip install -e '.[test]'` | 通过；基础/tiny CPU 测试依赖可用，pip check 无冲突 |
| 统一 CPU pytest | **105 通过、3 个模块跳过**（可选 vLLM 缺失），1 个预期 native cache 警告 |
| 分类收集 | base 44、HF 47、runtime 14；vLLM 单独收集两模块均跳过，退出码 5；GPU wrapper 收集 1 项但未执行 |
| 最小 CLI 与 tiny 模型 | demo_hf.py --help、公开 mini CLI、tiny 随机 HF forward 通过；未跑官方权重 demo |
| 公开 MMOU 准备 | 成功下载 2 视频／6 原题／834,580 媒体字节，manifest/回执哈希核验通过 |
| 公开推理/报告合约 | 用明确的模拟 scorer 单测，验证原标签保留、输入不泄漏参考、部分结果拒绝；**不是模型准确率** |
| 公开官方模型 HF/vLLM 推理 | **未执行**：本机无 NVIDIA CUDA 设备，本次未下载官方权重 |
| Linux 安装 / Docker 构建 | **未执行**：本机为 macOS ARM64，Docker daemon 不可用 |
| 完整媒体解码 / processor 预检 | **未执行**：无 FFmpeg 及完整 HF 音视频依赖 |

实际依赖见 [版本清单](validation_evidence/macos-arm64-python311-packages.json)。步骤及限制见 [安装验证](installation_validation.zh-CN.md)、[测试入口](testing.zh-CN.md)、[公开评测](reproduce.zh-CN.md)。下载完整性不代表完整解码，tiny 随机模型正确性不代表预训练准确率。

## 历史成绩：保留、不重新解释

- [无缓存配对评测](paired_backends.zh-CN.md)：原记录为 BF16 权重、HF 分层/vLLM TP4、串行/无缓存/完整读出，表格不改。精确模型 revision、完整环境锁定及显式 numerics 配置证据在公开仓库中**未核实**，不得把当前 stable 默认值套给旧成绩。
- [稳定数值回归](stability.zh-CN.md)：原记录 HF stable/vLLM invariant、27 题小规模数值对照，不是准确率。精确历史模型 revision 与完整环境锁仍未核实，本次未重跑 GPU。
- [native 缓存失败](hf.zh-CN.md)：保留失败验收，不能被 tiny CPU 通过静默替代。
- [早期数据审计](benchmark_validation.zh-CN.md)：仅数据完整性；新公开 mini 是不同抽样。

旧本地数据包 runner 新运行须显式 numerics/model commit，记录实际源码/依赖，并标注本地 revision 身份未核实。无 run.json 的历史报告为 unverified_historical_configuration。新公开流程直接解析固定 Hub revision；这些机制都不追溯认证历史产物。
