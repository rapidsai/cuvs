#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Generate C++ testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-logger "Create test environment"
rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Discovering libcuvs test executables"

# Navigate to test installation directory
TEST_DIR="${CONDA_PREFIX}/bin/gtests/libcuvs"

if [ ! -d "${TEST_DIR}" ]; then
  rapids-logger "Error: Test directory ${TEST_DIR} not found"
  exit 1
fi

cd "${TEST_DIR}"

# Find all *_TEST executables
if ! ls ./*_TEST 1> /dev/null 2>&1; then
  rapids-logger "Error: No test executables found matching *_TEST pattern"
  exit 1
fi

# Tests that exceed the 360-minute GitHub Actions job time limit, per tool.
# These are excluded from the matrix to avoid wasting CI resources.
MEMCHECK_SKIP="
  NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST
  NEIGHBORS_ANN_IVF_FLAT_TEST
  NEIGHBORS_ANN_IVF_PQ_TEST
  NEIGHBORS_ANN_VAMANA_TEST
  NEIGHBORS_DYNAMIC_BATCHING_TEST
"
RACECHECK_SKIP="
  CLUSTER_TEST
  DISTANCE_TEST
  NEIGHBORS_ALL_NEIGHBORS_TEST
  NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST
  NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST
  NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST
  NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST
  NEIGHBORS_ANN_IVF_FLAT_TEST
  NEIGHBORS_ANN_IVF_PQ_TEST
  NEIGHBORS_ANN_NN_DESCENT_TEST
  NEIGHBORS_ANN_SCANN_TEST
  NEIGHBORS_ANN_VAMANA_TEST
  NEIGHBORS_DYNAMIC_BATCHING_TEST
  NEIGHBORS_EPSILON_NEIGHBORHOOD_TEST
  NEIGHBORS_HNSW_TEST
"
SYNCCHECK_SKIP="
  NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST
  NEIGHBORS_ANN_NN_DESCENT_TEST
  NEIGHBORS_DYNAMIC_BATCHING_TEST
"

# Build a per-tool filtered JSON array of test names.
# Excludes multi-GPU tests (NEIGHBORS_MG_*) and per-tool timeout skips.
filter_tests() {
  local skip_list="$1"
  local filtered=()
  for test in "${all_tests[@]}"; do
    if [[ "${skip_list}" == *"${test}"* ]]; then
      continue
    fi
    filtered+=("$test")
  done
  printf '%s\n' "${filtered[@]}" | jq -R -s -c 'split("\n") | map(select(length > 0))'
}

all_tests=()
for test in *_TEST; do
  if [[ ! "$test" =~ ^NEIGHBORS_MG_ ]]; then
    all_tests+=("$test")
  fi
done

memcheck_tests=$(filter_tests "${MEMCHECK_SKIP}")
racecheck_tests=$(filter_tests "${RACECHECK_SKIP}")
synccheck_tests=$(filter_tests "${SYNCCHECK_SKIP}")

rapids-logger "memcheck tests:"
echo "${memcheck_tests}" | jq .[]
rapids-logger "racecheck tests:"
echo "${racecheck_tests}" | jq .[]
rapids-logger "synccheck tests:"
echo "${synccheck_tests}" | jq .[]

# Output to GITHUB_OUTPUT for GitHub Actions
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "memcheck_tests=${memcheck_tests}"
    echo "racecheck_tests=${racecheck_tests}"
    echo "synccheck_tests=${synccheck_tests}"
  } >> "${GITHUB_OUTPUT}"
fi
