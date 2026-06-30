#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare baseline / cuTile / flash-kmeans for one shape.
#
# Usage:
#   export BENCH_CONDA=/path/to/miniforge3
#   export BENCH_ENV_BASE=...
#   export BENCH_ENV_CUTILE=...
#   export BENCH_ENV_FLASH=...
#   export MAX_ITER=5 TOL=1e-4 SEED=42
#   export WARMUP_FIT=1 ITERS_FIT=3 WARMUP_PRED=1 ITERS_PRED=3
#   ./run_benchmark_kmeans.sh N D K
#

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 3 ]]; then
  echo "usage: $0 N D K" >&2
  echo "See script header for required env vars and examples." >&2
  exit 2
fi

: "${BENCH_CONDA:?set BENCH_CONDA to conda/miniforge root}"
: "${BENCH_ENV_BASE:?set BENCH_ENV_BASE}"
: "${BENCH_ENV_CUTILE:?set BENCH_ENV_CUTILE}"
: "${BENCH_ENV_FLASH:?set BENCH_ENV_FLASH}"
: "${MAX_ITER:?set MAX_ITER}"
: "${SEED:?set SEED}"
: "${WARMUP_FIT:?set WARMUP_FIT}"
: "${ITERS_FIT:?set ITERS_FIT}"
: "${WARMUP_PRED:?set WARMUP_PRED}"
: "${ITERS_PRED:?set ITERS_PRED}"
: "${TOL:?set TOL}"

N=$1
D=$2
K=$3

exec python3 "$SCRIPT_DIR/benchmark_kmeans.py" --compare \
  --n "$N" --d "$D" --k "$K" \
  --max-iter "$MAX_ITER" --tol "$TOL" --seed "$SEED" \
  --warmup-fit "$WARMUP_FIT" --iters-fit "$ITERS_FIT" \
  --warmup-pred "$WARMUP_PRED" --iters-pred "$ITERS_PRED"
