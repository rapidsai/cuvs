#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail


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

# [CI ONLY] Repeat the flaky IVF-Flat int8/uint8 cosine test.
BIN="$CONDA_PREFIX"/bin/gtests/libcuvs/NEIGHBORS_ANN_IVF_FLAT_TEST
repeat_test() {
  local name=$1 filter=$2 repeat=$3
  local log="${RAPIDS_TESTS_DIR}/${name}.log"
  rapids-logger "Repeat --gtest_filter=${filter} ${repeat} times"
  timeout -v --signal=SIGINT --kill-after=60s 150m "${BIN}" --gtest_filter="${filter}" --gtest_repeat="${repeat}" > "${log}" 2>&1
  echo "Exit status: $?"
  tail -n 5 "${log}"
  echo "Failure details:"
  grep -E -B3 -A6 "Failure$" "${log}" | head -n 400 || true
  echo "Diagnostics:"
  grep -E -A0 "DIAG|^  (naive|ivf|coarse)|Recall = 0\.[0-8]|Duplicated" "${log}" | head -n 2000 || true
  echo "Failures:"
  grep -E "^\[  FAILED  \] .* \([0-9]+ ms\)$" "${log}" | sed -E 's/, where GetParam.*//; s/ \([0-9]+ ms\)//' | sort | uniq -c || true
  echo "Passes:"
  grep -E "^\[       OK \]" "${log}" | sed -E 's/ \([0-9]+ ms\)//' | sort | uniq -c || true
}
nvidia-smi -q | grep -E "Driver Version|CUDA Version|Product Name" || true
rapids-logger "Zero-norm center regression test WITHOUT the guard (expected to fail)"
CUVS_CI_DISABLE_ZERO_NORM_GUARD=1 "${BIN}" --gtest_filter='AnnIVFFlatTest.CosineZeroNormCenters' 2>&1 | tail -n 20 || true
rapids-logger "Zero-norm center regression test WITH the guard"
"${BIN}" --gtest_filter='AnnIVFFlatTest.CosineZeroNormCenters' --gtest_repeat=20 2>&1 | grep -E "Duplicated|Recall|FAILED|PASSED|OK \]" | sort | uniq -c
repeat_test cosine2 '*AnnIVFFlatTestF_int8.AnnIVFFlat/2:*AnnIVFFlatTestF_uint8.AnnIVFFlat/2' 800
repeat_test full_ivf_flat '*' 1

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
