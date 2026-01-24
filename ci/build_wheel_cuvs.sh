#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Only use stable ABI package for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  source ./ci/use_upstream_sabi_wheels.sh
fi

package_dir="python/cuvs"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcuvs wheels from this current build,
# then ensures 'cuvs' wheel builds always use the 'libcuvs' just built in the same CI run.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
LIBCUVS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUVS_WHEELHOUSE}"/libcuvs_*.whl)" >> "${PIP_CONSTRAINT}"

ci/build_wheel.sh cuvs ${package_dir}
ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RAPIDS_PACKAGE_NAME="$(rapids-package-name wheel_python cuvs --stable --cuda)"
  export RAPIDS_PACKAGE_NAME
fi
