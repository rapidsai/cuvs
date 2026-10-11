#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Usage: ci/test_cpp.sh [SHARD NUM_SHARDS]
# With SHARD and NUM_SHARDS (1 <= SHARD <= NUM_SHARDS), only every NUM_SHARDS-th libcuvs test is
# run, starting from test number SHARD, so the tests can be split across several CI jobs.

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

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# [DO NOT MERGE] Workstream Q: verify the multi-partition fix by repeating every AnnCagraMultiPartitionTest
# case for every data type, and the num_partitions=1 cases with a different dataset per repetition. The
# recall of every repetition is logged as a QREC line (run, test, recall).
rapids-logger "Workstream Q: AnnCagraMultiPartitionTest verification"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
GT="$CONDA_PREFIX"/bin/gtests/libcuvs
QLOG="${RAPIDS_TESTS_DIR}/qdiag"
mkdir -p "${QLOG}"
NP1=""
for i in 0 1 2 3 4 5; do
  NP1="${NP1}AnnCagraMultiPartitionTest/*.Search/${i}:AnnCagraMultiPartitionTest/*.FilteredSearch/${i}:"
done
ALLMP='AnnCagraMultiPartitionTest/*'

run_q() {
  local name=$1
  shift
  rapids-logger "Q run: ${name}: $*"
  "$@" > "${QLOG}/${name}.log" 2>&1
  echo "exit code: $?"
  grep -E '^\[  FAILED  \] Ann' "${QLOG}/${name}.log" | sort | uniq -c
  awk -v run="${name}" '
    /^\[ RUN      \] / { t = $4 }
    /Recall = / { for (i = 1; i <= NF; i++) if ($i == "=") { r = $(i + 1); break }
                  printf "QREC %s %s %s\n", run, t, r }' "${QLOG}/${name}.log" > "${QLOG}/${name}.rec"
  awk '{ n++; if (!(1 in mn) || $4 < mn[1]) mn[1] = $4 }
       END { printf "QSUM %s n=%d min=%.3f\n", "'"${name}"'", n, mn[1] }' "${QLOG}/${name}.rec"
}

for b in FLOAT HALF INT8 UINT8; do
  run_q "all_${b}" "${GT}/NEIGHBORS_ANN_CAGRA_${b}_UINT32_TEST" --gtest_filter="${ALLMP}" --gtest_repeat=40
done
for b in HALF INT8 UINT8; do
  CUVS_QDIAG_SEED_SWEEP=1 run_q "np1_sweep_${b}" "${GT}/NEIGHBORS_ANN_CAGRA_${b}_UINT32_TEST" --gtest_filter="${NP1}" --gtest_repeat=150
done

rapids-logger "Q raw QREC lines"
cat "${QLOG}"/*.rec

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
