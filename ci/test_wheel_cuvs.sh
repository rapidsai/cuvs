#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

# Delete system libnccl.so to ensure the wheel is used
rm -rf /usr/lib64/libnccl*

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUVS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
CUVS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    "${LIBCUVS_WHEELHOUSE}"/libcuvs*.whl \
    "$(echo "${CUVS_WHEELHOUSE}"/cuvs*.whl)[test]"

python -m pytest ./python/cuvs/cuvs/tests
