[English](clean_linux_validation.md) · **简体中文**

# 干净 Linux 公开流程验证：2026-09-26

已按公开 HF 教程完整执行 **数据准备 → 官方模型真实推理 → 报告生成**，覆盖 MMOU 两段公开视频和全部六道原题。这验证了指定环境和配置可运行，不代表任意硬件保证、官方榜单成绩或历史实验复现。

## 验证环境与固定配置

- Linux x86_64、Debian 13 容器、Python 3.11.16；四张 24 GB NVIDIA GPU，驱动 580.105.08。
- PyTorch 2.11.0+cu130、Transformers 5.13.1、Torchvision 0.26.0+cu130、TorchCodec 0.11.0+cpu；系统 FFmpeg 7.1.5。[环境与检查记录](validation_evidence/linux-clean-install.json)；[全部 Python 依赖版本及源码哈希](validation_evidence/linux-public-mmou-hf-report.json)。
- 空白 venv，不继承系统或旧环境已安装包。复用了 pip 下载缓存及独立核验的官方权重缓存；不是从网络重新下载全部约 70.5 GB 权重。
- 官方模型 revision：`26291f793822fb6be9555850f06dfe95f2d7e695`。15 个分片均与固定 Hub LFS 元数据的 SHA256 一致：[权重回执](validation_evidence/linux-official-weight-integrity.json)。runner 本身记录 Hub 来源，逐分片校验是另行执行的补充证据。
- 代码基线 `04cc2e0a8fd723d678f44b0c69c2e74b4fe94ea4`，加本次安装及来源记录修复；报告记录实际源码哈希。工作区为 dirty，不声称未修改的原 commit 已通过。
- 使用 `configs/public_mini_hf.json`：stable、causal、BF16 权重、完整 LM Head 投影、batch=1、无 prefix cache、4,000 token 上限、seed=37、官方 processor；解码后端明确为 `torchcodec`。没有修改权重、attention 或候选评分逻辑。

按 [HF 安装教程](hf.zh-CN.md)先安装官方 CPU 解码包，再执行[公开复现流程](reproduce.zh-CN.md)中的命令。CPU 索引只用于解码包，模型推理仍使用 CUDA PyTorch；无需安装 vLLM 或编译自定义 CUDA 组件。

## 实际结果

| 检查 | 结果 |
| --- | --- |
| 干净安装 / `pip check` | 通过 |
| 固定 seed 的 CUDA 矩阵运算 | 四张 GPU 均通过 |
| CPU pytest | 105 通过，3 个可选 vLLM 模块跳过 |
| 原视频解码 | H.264、AV1 两段均通过，未转码 |
| 公开准备 / 推理 / 报告 | 2 段视频、6/6 题，三个完成标记均生成 |
| 实际完整输入长度 | 920–1,255 token，低于 4,000 |
| 图片 demo `--check-only` | 通过；119 token，A/B/C token ID 为 32/33/34；不是图片完整模型推理 |

六道原始标签题的结果：**2/6 正确（33.33%）**，NLL **1.346751**，多分类 Brier **0.703255**，ECE **0.327820**（15 bins）。这些小样本指标用于验证报告链路，不能据此评价整体准确率或校准质量。[机器可读报告](validation_evidence/linux-public-mmou-hf-report.json)。

模型加载 **25.70 秒**。两组媒体分别有 2、4 道题，耗时 **24.63 秒**和 **16.65 秒**，评测合计 **41.28 秒**。组耗时包含解码、processor 和评分，无预热；不包含模型加载及 Hub 解析。不能将其当作单题 API 延迟或并发性能。

## 保留的失败与修复

1. Python 3.10 无法安装固定 PyAV 18.1.0；包声明和教程统一改为 Python 3.11+。
2. 早期另一环境出现 PyPI 下载超时，没有计作安装成功。
3. 原 HF extra 缺少官方视频 processor 所需的 Torchvision；现固定匹配版本 0.26.0。
4. Decord 无法读取选中的 AV1 视频，回退又调用新版 Torchvision 已移除的 `read_video`。该轮只完成 2/6 题，没有成功报告。保留原样本，改用官方工具已有的 TorchCodec 解码路径。
5. 默认 TorchCodec 包依赖缺失的 CUDA 视频库；最终干净安装明确使用官方 CPU 解码包，模型仍在 GPU 推理，并记录解码后端及依赖版本。

失败轮次的日志和产物分别保存，没有混入最终成绩。最终文档命令在已有安装环境中再次执行。完整日志和原始输出保留在 Git 外；公开证据只包含配置、版本、指标和哈希，不含私有服务器路径、媒体、标注或权重。

## 剩余范围

仅完整验证一组 HF 配置、一次运行。未新增 vLLM、HTTP/Tree-KV、cache/batch 一致性、独立图片完整模型或全量 benchmark 结论；历史成绩的配置及未核实状态保持原样。统一入口见[最新验证](validation_current.zh-CN.md)。
