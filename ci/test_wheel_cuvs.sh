#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cuvs*.whl)[test]

# arm tests sporadically run into https://bugzilla.redhat.com/show_bug.cgi?id=1722181.
# This is a workaround to ensure that OpenMP gets the TLS that it needs.
if [[ "$(arch)" == "aarch64" ]]; then
  for dir in $(ld --verbose | grep SEARCH_DIR | tr -s ' ;' \\012 | grep -o '=.*"' | sed 's/[="]//g'); do
      lib="${dir}/libgomp.so.1"
      if [[ -f "${lib}" ]]; then
          export LD_PRELOAD="${lib}"
          break
      fi
  done
fi

python -m pytest ./python/cuvs/cuvs/test
