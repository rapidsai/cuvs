#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libcuvs-dep
RAPIDS_PY_WHEEL_NAME="cuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

# TODO: remove me once gha-tools is released
if ! command -v "rapids-pip-retry" &> /dev/null
then
    git clone --branch rapids-pip-retry https://github.com/gforsyth/gha-tools.git

    export PATH="$PWD/gha-tools/tools":$PATH
fi

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    ./local-libcuvs-dep/libcuvs*.whl \
    "$(echo ./dist/cuvs*.whl)[test]"

python -m pytest ./python/cuvs/cuvs/tests
