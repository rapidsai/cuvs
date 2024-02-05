#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

## echo to expand wildcard before adding `[extra]` requires for pip
#python -m pip install $(echo ./dist/cuvs*.whl)[test]
#
## Run smoke tests for aarch64 pull requests
#if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
#    python ./ci/wheel_smoke_test_cuvs.py
#else
#    python -m pytest ./python/cuvs/cuvs/test
#fi
