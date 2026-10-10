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

# Run the k-means and IVF test binaries once in full.
rapids-logger "Run k-means and IVF test binaries"
pushd "$CONDA_PREFIX"/bin/gtests/libcuvs
timeout -v --signal=SIGINT --kill-after=60s 120m ctest -j4 --output-on-failure -R "^(CLUSTER_TEST|CLUSTER_KMEANS_MG_TEST|CLUSTER_KMEANS_MNMG_TEST|NEIGHBORS_ANN_IVF_FLAT_TEST|NEIGHBORS_ANN_IVF_PQ_TEST|NEIGHBORS_ANN_IVF_SQ_TEST|NEIGHBORS_ANN_IVF_RABITQ_TEST)$"
popd

# Repeat the reproducibility tests.
repeat_test() {
  local binary=$1 filter=$2 repeat=$3
  local log="$CONDA_PREFIX/bin/gtests/libcuvs/${binary}_repeat.log"
  rapids-logger "Repeat ${binary} --gtest_filter=${filter} ${repeat} times"
  "$CONDA_PREFIX"/bin/gtests/libcuvs/"${binary}" --gtest_filter="${filter}" --gtest_repeat="${repeat}" > "${log}" 2>&1
  echo "Exit status: $?"
  tail -n 5 "${log}"
  grep -E "Failure|differ|Expected" "${log}" | head -n 50 || true
  echo "Failures:"
  grep -E "^\[  FAILED  \] .* \([0-9]+ ms\)$" "${log}" | sed -E 's/, where GetParam.*//; s/ \([0-9]+ ms\)//' | sort | uniq -c || true
  echo "Passes:"
  grep -E "^\[       OK \]" "${log}" | sed -E 's/ \([0-9]+ ms\)//' | sort | uniq -c || true
}
repeat_test CLUSTER_TEST "KmeansFitBatchedTests/*:KmeansBalancedReproducible.*" 50
repeat_test NEIGHBORS_ANN_IVF_FLAT_TEST "IVFFlatBuild.Reproducible" 10
repeat_test NEIGHBORS_ANN_IVF_PQ_TEST "IvfPqBuild.Reproducible*" 10

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
