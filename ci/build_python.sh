#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

package_name="cuvs"
package_dir="python"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
PYLIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 python conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)
PYLIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 python conda)

version=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION=${version}
echo "${version}" > VERSION

sccache --zero-stats

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cuvs

sccache --show-adv-stats
sccache --zero-stats

# Build cuvs-bench for each cuda and python version
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cuvs-bench

sccache --show-adv-stats
sccache --zero-stats

# Build cuvs-bench-cpu only in CUDA 12 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "12" ]]; then
  rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cuvs-bench-cpu

  sccache --show-adv-stats
fi

rapids-upload-conda-to-s3 python
