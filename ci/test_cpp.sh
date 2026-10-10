#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Usage: ci/test_cpp.sh [SHARD NUM_SHARDS]
# With SHARD and NUM_SHARDS (1 <= SHARD <= NUM_SHARDS), only every NUM_SHARDS-th libcuvs test is
# run, starting from test number SHARD, so the tests can be split across several CI jobs.
export SHARD=${1:-1}
export NUM_SHARDS=${2:-1}

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

# CI provides CUDA_CACHE_PATH through the reusable workflow's cache-environment input.
# So that we can re-use the CUDA driver's on-disk JIT cache between runs.
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-.cache/cuda-jit}"
if [[ "${CUDA_CACHE_PATH}" != /* ]]; then
  CUDA_CACHE_PATH="$(realpath -m "${CUDA_CACHE_PATH}")"
fi
export CUDA_CACHE_PATH
mkdir -p "${CUDA_CACHE_PATH}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
RAPIDS_DATASET_ROOT_DIR=${RAPIDS_TESTS_DIR}/dataset
export RAPIDS_DATASET_ROOT_DIR
# (test data download skipped for the MG repro loop)

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# [DO NOT MERGE] Reproduce flaky NEIGHBORS_MG_TEST (CAGRA SHARDED/ROUND_ROBIN on a single GPU).
# Run the affected cases repeatedly in fresh processes for a fixed time budget and count aborts.
rapids-logger "Repeat NEIGHBORS_MG_TEST AnnMG/5, /8, /14"
nvidia-smi -L
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
MG_FILTER='AnnMGTest/AnnMGTestF_float.AnnMG/5:AnnMGTest/AnnMGTestF_float.AnnMG/8:AnnMGTest/AnnMGTestF_float.AnnMG/14'
END=$((SECONDS + 50 * 60))
N_PROC=0
N_FAIL=0
N_OK=0
while [[ ${SECONDS} -lt ${END} ]]; do
  N_PROC=$((N_PROC + 1))
  timeout -v --signal=SIGINT --kill-after=60s 20m ./NEIGHBORS_MG_TEST --gtest_filter="${MG_FILTER}" --gtest_repeat=20 > mg_run.log 2>&1
  rc=$?
  ok=$(grep -c 'OK \]' mg_run.log)
  N_OK=$((N_OK + ok))
  if [[ ${rc} -ne 0 ]]; then
    N_FAIL=$((N_FAIL + 1))
    echo "=== process ${N_PROC} FAILED (rc=${rc}) after ${ok} passing tests; last test started:"
    grep 'RUN ' mg_run.log | tail -1
    grep -v -E 'Recall =|^\[ *(RUN|OK) |^Note:|^Repeating|^$' mg_run.log | tail -40
  fi
done
popd
echo "MG SUMMARY: ${N_FAIL}/${N_PROC} processes failed, ${N_OK} tests passed"
if [[ ${N_FAIL} -gt 0 ]]; then EXITCODE=1; fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
