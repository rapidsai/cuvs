#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Delete system libnccl.so to ensure the wheel is used
rm -rf /usr/lib64/libnccl*

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBCUVS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
CUVS_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" cuvs --stable --cuda "$RAPIDS_CUDA_VERSION")")

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    "${LIBCUVS_WHEELHOUSE}"/libcuvs*.whl \
    "$(echo "${CUVS_WHEELHOUSE}"/cuvs*.whl)[test]"

for i in $(seq 1 10); do
  echo "===== Run $i of 10 ====="
  PYTHONUNBUFFERED=1 python -m pytest -vs ./python/cuvs/cuvs/tests
done
