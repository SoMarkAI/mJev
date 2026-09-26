[English](README.md) · **简体中文**

# 实验音视频镜像

这是可选的 vLLM 镜像，额外安装音视频处理依赖，供历史音视频评测和实验 HTTP runtime 使用。默认 HF 快速开始不需要它，它也不是 HF 专用镜像。

请在**仓库根目录**构建，保留 `.` 作为构建上下文：

```bash
docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .
```

默认入口为 `python3 -m mjev.benchmark.pilot`，启动的是评测程序而非 HTTP 服务。输入与运行方法见[音视频评测指南](../../docs/av_pilot.zh-CN.md)；[实验 runtime 指南](../../docs/integrated.zh-CN.md)中的启动脚本会覆盖此入口。

普通 vLLM demo 镜像使用根目录 [Dockerfile](../../Dockerfile)，参见 [vLLM 部署指南](../../docs/vllm.zh-CN.md)。两个镜像使用同一个固定版本的 vLLM 基础镜像。此次从根目录移动文件不改变依赖和启动入口，已有 `mjev-av-pilot:0.1` 镜像仍可使用。
