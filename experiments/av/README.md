**English** &nbsp;&nbsp; [简体中文](README.zh-CN.md)

# Experimental audio/video image

This optional vLLM image adds media-processing dependencies for the historical AV pilot and experimental HTTP runtime. It is not required by the default HF Quick Start and is not an HF-specific image.

Build **from the repository root**, keeping `.` as the build context:

```bash
docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .
```

The default entry point is `python3 -m mjev.benchmark.pilot`, an evaluation program rather than an HTTP server. See the [AV pilot guide](#pilot) for inputs and execution, or the [experimental runtime guide](../../docs/integrated.md), whose launcher overrides the entry point.

## Historical environment tools

- [av_witness.py](av_witness.py) runs a seeded BF16 SDPA kernel check on exactly four GPUs. It checks the environment rather than model inference.
- [av_env.json](av_env.json) describes the historical four-GPU pilot environment; the inference engines do not read it as runtime configuration.

With four available GPUs, run from the repository root:

```bash
docker run --rm --gpus all --entrypoint python3 mjev-av-pilot:0.1 experiments/av/av_witness.py
```

For the ordinary vLLM demo image, use the root [Dockerfile](../../Dockerfile) and [vLLM deployment guide](../../docs/vllm.md). Both images use the same pinned vLLM base. Rebuild the AV image to use the reorganized tool paths; its dependencies and evaluation entry point are unchanged.


<a id="pilot"></a>

## Run the optional pilot

Set `MODEL_DIR` to local official weights and `BENCH_ROOT` to a prepared AV bundle. This is an experimental workflow, not a current benchmark claim.

```bash
mkdir -p outputs/av-pilot
docker run --rm -v "$MODEL_DIR:/model:ro" -v "$BENCH_ROOT:/bench:ro" \
  -v "$PWD/outputs/av-pilot:/out" mjev-av-pilot:0.1 \
  --model /model --root /bench --output /out/preflight --check-only
```

With four free GPUs, add `--gpus all --ipc=host -e VLLM_BATCH_INVARIANT=0`, omit `--check-only`, and pass `--numerics native --model-revision "$MODEL_REVISION"`. Kernel checks and inference are distinct; a successful CPU preflight does not validate GPU accuracy. Media is not silently truncated. Use a fresh output directory for a new run.
