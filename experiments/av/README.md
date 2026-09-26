**English** · [简体中文](README.zh-CN.md)

# Experimental audio/video image

This optional vLLM image adds media-processing dependencies for the historical AV pilot and experimental HTTP runtime. It is not required by the default HF Quick Start and is not an HF-specific image.

Build **from the repository root**, keeping `.` as the build context:

```bash
docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .
```

The default entry point is `python3 -m mjev.benchmark.pilot`, an evaluation program rather than an HTTP server. See the [AV pilot guide](../../docs/av_pilot.md) for inputs and execution, or the [experimental runtime guide](../../docs/integrated.md), whose launcher overrides the entry point.

For the ordinary vLLM demo image, use the root [Dockerfile](../../Dockerfile) and [vLLM deployment guide](../../docs/vllm.md). Both images use the same pinned vLLM base. Moving this file from the root does not change its dependencies or entry point; existing `mjev-av-pilot:0.1` images remain usable.
