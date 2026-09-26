#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MJEV_MODEL:?Set MJEV_MODEL to the official local model directory}"
export PYTHONPATH="$PWD:$PWD/runtime${PYTHONPATH:+:$PYTHONPATH}"
export MJEV_ENABLE=1 MJEV_ENABLE_PATCHES=0 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
exec "${PYTHON:-python}" -m pytest tests --suite gpu -ra "$@"
