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

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints py_test_cuvs ./constraints.txt

# notes:
#
#   * echo to expand wildcard before adding `[test]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
    --constraint ./constraints.txt \
    --constraint "${PIP_CONSTRAINT}" \
    "${LIBCUVS_WHEELHOUSE}"/libcuvs*.whl \
    "$(echo "${CUVS_WHEELHOUSE}"/cuvs*.whl)[test]"

python -m pytest ./python/cuvs/cuvs/tests
