#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
PYLIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 python conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)
PYLIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 python conda)

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  "libcuvs=${RAPIDS_VERSION}" \
  "cuvs=${RAPIDS_VERSION}"

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuvs"
pushd python/cuvs/cuvs
pytest \
 --cache-clear \
 --junitxml="${RAPIDS_TESTS_DIR}/junit-cuvs.xml" \
 --cov-config=../.coveragerc \
 --cov=cuvs \
 --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuvs-coverage.xml" \
 --cov-report=term \
 tests

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
