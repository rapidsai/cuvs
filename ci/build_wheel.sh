#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/wheel/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

rapids-generate-version > ./VERSION

cd "${package_dir}"

EXCLUDE_ARGS=(
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnccl.so.*"
  --exclude "libnvJitLink.so.*"
  --exclude "libraft.so"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)

if [[ "${package_dir}" != "python/libcuvs" ]]; then
    EXCLUDE_ARGS+=(
      --exclude "libcuvs_c.so"
      --exclude "libcuvs.so"
    )
fi

SKBUILD_CMAKE_ARGS="-DUSE_NCCL_RUNTIME_WHEEL=ON"
export SKBUILD_CMAKE_ARGS

rapids-print-env

rapids-logger "Building '${package_name}' wheel"

sccache --stop-server 2>/dev/null || true

rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" "${EXCLUDE_ARGS[@]}" dist/*
