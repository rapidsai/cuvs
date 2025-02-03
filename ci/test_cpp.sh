#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  "libcuvs=${RAPIDS_VERSION}" \
  "libcuvs-tests=${RAPIDS_VERSION}"

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run libcuvs gtests from libcuvs-tests package
cd "$CONDA_PREFIX"/bin/gtests/libcuvs
ctest -j8 --output-on-failure

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
