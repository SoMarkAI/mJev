#!/usr/bin/env bash
set -euo pipefail
# Build the default image from the repository root:
# docker build -f experiments/av/Dockerfile -t mjev-av-pilot:0.1 .
: "${MODEL_DIR:?Set MODEL_DIR}" "${DATA_DIR:?Set DATA_DIR}"
root="$(cd "$(dirname "$0")/.." && pwd)"
docker run -d --name mjev-integrated --gpus all --ipc=host \
 -p 127.0.0.1:17005:8000 -w /workspace/runtime \
 -v "$root:/workspace:ro" -v "$MODEL_DIR:/model:ro" -v "$DATA_DIR:/data:ro" \
 -e PYTHONPATH=/workspace/runtime:/workspace -e MJEV_ENABLE_PATCHES=1 \
 -e MJEV_ENABLE=0 -e VLLM_USE_V2_MODEL_RUNNER=0 \
 -e MJEV_ALLOWED_MEDIA_ROOTS=/data \
 -e MJEV_ALLOWED_MEDIA_HOSTS="${MJEV_ALLOWED_MEDIA_HOSTS:-}" \
 -e MJEV_MAX_MEDIA_BYTES="${MJEV_MAX_MEDIA_BYTES:-536870912}" \
 --entrypoint python3 "${MJEV_IMAGE:-mjev-av-pilot:0.1}" /workspace/runtime/mjev_server.py \
 --model /model --served-model-name Qwen3-Omni-30B-A3B-Instruct \
 --host 0.0.0.0 --port 8000 --dtype bfloat16 --tensor-parallel-size 4 \
 --gpu-memory-utilization 0.9 --max-model-len 4096 --max-num-seqs 32 \
 --max-num-batched-tokens 4096 --enable-prefix-caching --block-size 16 \
 --max-logprobs 27 --no-async-scheduling --enforce-eager \
 --allowed-local-media-path /data --limit-mm-per-prompt '{"image":1,"video":1,"audio":1}'
