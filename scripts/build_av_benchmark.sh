#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BENCH_ROOT:?Set BENCH_ROOT to a data directory outside the checkout}"
"${PYTHON:-python}" -m mjev.benchmark.prepare --output "$BENCH_ROOT" --seed "${BENCH_SEED:-20260925}"
"${PYTHON:-python}" -m mjev.benchmark.fetch --root "$BENCH_ROOT" --workers "${BENCH_WORKERS:-8}"
"${PYTHON:-python}" -m mjev.benchmark.audit --root "$BENCH_ROOT"
