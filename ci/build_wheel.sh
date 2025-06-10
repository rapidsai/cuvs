#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string
source rapids-init-pip

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

rapids-logger "Building '${package_name}' wheel"

sccache --zero-stats

rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" "${EXCLUDE_ARGS[@]}" dist/*
