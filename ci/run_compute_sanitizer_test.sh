#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

# This script runs compute-sanitizer on a single libcuvs test executable
# Usage: ./run_compute_sanitizer_test.sh TOOL_NAME TEST_NAME [additional gtest args...]
# Example: ./run_compute_sanitizer_test.sh memcheck DISTANCE_TEST
# Example: ./run_compute_sanitizer_test.sh racecheck CLUSTER_TEST --gtest_filter=KMeans.*

if [ $# -lt 2 ]; then
  echo "Error: Tool and test name required"
  echo "Usage: $0 TOOL_NAME TEST_NAME [additional gtest args...]"
  echo "  TOOL_NAME: compute-sanitizer tool (memcheck, racecheck, synccheck)"
  echo "  TEST_NAME: libcuvs test name"
  exit 1
fi

TOOL_NAME="${1}"
shift
TEST_NAME="${1}"
shift

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

rapids-logger "Check GPU usage"
nvidia-smi

# Download test data required by NEIGHBORS_ANN_VAMANA_TEST
RAPIDS_DATASET_ROOT_DIR="$(mktemp -d)"
export RAPIDS_DATASET_ROOT_DIR
./ci/get_test_data.sh --NEIGHBORS_ANN_VAMANA_TEST

rapids-logger "Running compute-sanitizer --tool ${TOOL_NAME} on ${TEST_NAME}"

# Allows tests to detect they are running under compute-sanitizer
export "CUVS_${TOOL_NAME^^}_ENABLED=1"

# Navigate to test installation directory
TEST_DIR="${CONDA_PREFIX}/bin/gtests/libcuvs"
TEST_EXECUTABLE="${TEST_DIR}/${TEST_NAME}"

if [ ! -x "${TEST_EXECUTABLE}" ]; then
  rapids-logger "Error: Test executable ${TEST_EXECUTABLE} not found or not executable"
  exit 1
fi

CS_EXCLUDE_NAMES="kns=_no_sanitize,kns=_no_${TOOL_NAME}"

# Run compute-sanitizer on the specified test
compute-sanitizer \
  --tool "${TOOL_NAME}" \
  --force-blocking-launches \
  --kernel-name-exclude "${CS_EXCLUDE_NAMES}" \
  --track-stream-ordered-races all \
  --error-exitcode=1 \
  "${TEST_EXECUTABLE}" \
  "$@"

EXITCODE=$?

unset "CUVS_${TOOL_NAME^^}_ENABLED"

rapids-logger "compute-sanitizer --tool ${TOOL_NAME} on ${TEST_NAME} exiting with value: $EXITCODE"
exit $EXITCODE
