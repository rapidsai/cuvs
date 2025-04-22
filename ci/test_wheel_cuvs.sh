#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

# Delete system libnccl.so to ensure the wheel is used
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} != "11" ]]; then
  rm -rf /usr/lib64/libnccl*
fi

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libcuvs-dep
RAPIDS_PY_WHEEL_NAME="cuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    ./local-libcuvs-dep/libcuvs*.whl \
    "$(echo ./dist/cuvs*.whl)[test]"

python -m pytest ./python/cuvs/cuvs/tests
