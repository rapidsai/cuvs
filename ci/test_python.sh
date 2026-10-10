#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DO NOT MERGE: candidate-fix check for the H100 CAGRA Python test hang (#2772, #2732, #2783).
# Uses the packages built in this PR, repeats the PQ + CAGRA reproducer in separate processes,
# keeps going after a hang (each iteration is a new process), and then runs the full cuvs
# Python suite once.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cuvs cuvs --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "System info"
nvidia-smi
nvidia-smi -q | grep -iE "Addressing Mode|Driver Version|Product Name|VBIOS" || true
hostname || true
echo "RUNNER_NAME=${RUNNER_NAME:-}"
conda list | grep -E "^(libcuvs|cuvs) " || true

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"
DIAG_DIR="${PWD}/hang-diag"
mkdir -p "${DIAG_DIR}"

HANGS=0
# run_watched <name> <stall_seconds> <cmd...>: run a command, kill it if its output stalls.
run_watched() {
  local name=$1 stall=$2
  shift 2
  local log="${DIAG_DIR}/${name}.log"
  local start
  start=$(date +%s)
  echo ">>>>> START ${name}: $*"
  "$@" > "${log}" 2>&1 &
  local pid=$!
  local last_size=-1 last_change now size hung=0
  last_change=$(date +%s)
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 5
    size=$(stat -c %s "${log}" 2>/dev/null || echo 0)
    now=$(date +%s)
    if [[ "${size}" != "${last_size}" ]]; then
      last_size=${size}
      last_change=${now}
    elif (( now - last_change > stall )); then
      hung=1
      break
    fi
  done
  local rc=0
  if (( hung )); then
    HANGS=$((HANGS + 1))
    HANG_ITERS="${HANG_ITERS:-} ${name}"
    echo "!!!!! HANG DETECTED in ${name} after $(( $(date +%s) - start ))s (no output for ${stall}s)"
    tail -n 30 "${log}"
    nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv || true
    pkill -9 -P "${pid}" 2>/dev/null || true
    kill -9 "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null
    rc=124
  else
    wait "${pid}"
    rc=$?
    tail -n 3 "${log}"
  fi
  echo "<<<<< END ${name}: rc=${rc} elapsed=$(( $(date +%s) - start ))s"
  return ${rc}
}

set +e

CUVS_TESTS_DIR="${PWD}/python/cuvs/cuvs"
PYTEST_ARGS=(-v -p no:cacheprovider -o faulthandler_timeout=240)

N_ITERS=${N_ITERS:-10}
pushd "${CUVS_TESTS_DIR}" > /dev/null || exit 1
i=0
for i in $(seq 1 "${N_ITERS}"); do
  rapids-logger "Iteration ${i}/${N_ITERS} (hangs so far: ${HANGS})"
  run_watched "iter${i}-pq-cagra" 480 \
    python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" \
    tests/test_product_quantizer.py tests/test_cagra.py
  echo "ITERATION_RESULT iter=${i} total_hangs=${HANGS} hang_iters:${HANG_ITERS:-}"
  du -sh "${HOME}/.nv/ComputeCache" "${HOME}/.cupy/kernel_cache" 2>/dev/null || true
done
rapids-logger "REPRO SUMMARY: ${HANGS} hang(s) in ${i} iteration(s)"

rapids-logger "REPRO HANG ITERS:${HANG_ITERS:-}"
if (( HANGS == 0 )); then
  rapids-logger "Full cuvs Python suite"
  run_watched "full-suite" 900 \
    python -X faulthandler -m pytest "${PYTEST_ARGS[@]}" \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuvs.xml" tests
  FULL_RC=$?
  rapids-logger "FULL SUITE rc=${FULL_RC}"
fi
popd > /dev/null || exit 1

if (( HANGS > 0 )); then
  exit 1
fi
exit "${FULL_RC:-0}"
