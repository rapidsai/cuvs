#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
package_type=$3
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-generate-version > ./VERSION

cd "${package_dir}"

EXCLUDE_ARGS=(
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink.so.*"
)

if [[ "${package_dir}" != "python/libcuvs" ]]; then
    EXCLUDE_ARGS+=(
      --exclude "libcuvs_c.so"
      --exclude "libcuvs.so"
    )
fi

rapids-logger "Building '${package_name}' wheel"

sccache --zero-stats

rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair -w final_dist "${EXCLUDE_ARGS[@]}" dist/*

RAPIDS_PY_WHEEL_NAME="${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_type} final_dist
