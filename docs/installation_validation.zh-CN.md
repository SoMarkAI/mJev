[English](installation_validation.md) · **简体中文**

# 隔离环境安装验证：2026-09-26

**最新进展：**后续[干净 Linux 官方模型验证](clean_linux_validation.zh-CN.md)已在修复 Python／媒体依赖后跑通公开六题。以下保留早期 macOS 尝试的失败和范围，不代表当前 Linux 状态。

本机 macOS ARM64、Python 3.11；完整模型部署目标仍为 Linux/NVIDIA。新 venv 未启用 --system-site-packages，不继承既有环境包。

## 实际命令与结果

```bash
python3.11 -m venv /tmp/mjev-clean-20260926
/tmp/mjev-clean-20260926/bin/python -m pip install --retries 0 -e '.[hf]' pytest requests pyarrow
```

**失败**，停在依赖解析：`No matching distribution found for decord; extra == "decord"`。不能宣称 HF extra 安装成功；这也不证明 Linux 安装失败，本次没有验证 Linux。

随后独立验证 CPU 机制测试：

```bash
/tmp/mjev-clean-20260926/bin/python -m pip install --retries 0 -e '.[test]'
/tmp/mjev-clean-20260926/bin/python -m pip check
PYTHON=/tmp/mjev-clean-20260926/bin/python bash scripts/test.sh -q
/tmp/mjev-clean-20260926/bin/python demo_hf.py --help
/tmp/mjev-clean-20260926/bin/python -m mjev.benchmark.public_mini --help
```

最终安装和 pip check 通过；**105 项 CPU 测试通过，3 个模块因缺少 vLLM 跳过**。首轮暴露 accelerate 缺失及 runtime 可选 vLLM 依赖；test extra 现安装 accelerate，可选依赖缺失显式 skip。未放宽数值断言，未将模拟模型当作官方权重。

最小 forward 为 tiny 随机 Thinker 测试，覆盖 checkpoint 加载、模态、mask、cache。--help 仅验证 CLI 解析。新公开数据准备成功下载核验 2 视频／6 原题／834,580 媒体字节，不依赖内部包，见 [公开流程](reproduce.zh-CN.md)。

本环境核心版本 torch 2.11.0、Transformers 5.13.1、pytest 8.4.2、accelerate 1.15.0。[完整依赖清单](validation_evidence/macos-arm64-python311-packages.json) 是证据，**不是可移植 Linux 锁文件**。run.json 记录实际环境；以后重装时，未固定的传递依赖可能变化。

## 未执行

- 完整 HF 音视频安装、官方 processor 预检、官方 30B demo。
- 官方模型公开推理/准确率报告、vLLM GPU 回归。
- Docker 构建和全新 Linux 安装：本机非 Linux，Docker daemon 不可用。
- 完整音视频解码：没有 FFmpeg 或完整 AV 依赖。

在宣称全新 Linux 可用或发布新模型成绩前，应在全新 Linux/NVIDIA 环境实际跑安装与公开流程，保留依赖和运行证据；失败如实记录，不能将本次 CPU 验证替代 GPU 成功。
