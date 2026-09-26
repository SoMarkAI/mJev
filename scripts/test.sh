#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/runtime:$PWD/benchmarks/integrated${PYTHONPATH:+:$PYTHONPATH}"
export MJEV_ENABLE=0 MJEV_ENABLE_PATCHES=0 VLLM_USE_V2_MODEL_RUNNER=0
exec "${PYTHON:-python}" -m pytest tests -ra "$@"
