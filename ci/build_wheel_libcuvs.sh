#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="libcuvs"
package_dir="python/libcuvs"

rapids-logger "Generating build requirements"
matrix_selectors="cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "${matrix_selectors}" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
rapids-pip-retry install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

ci/build_wheel.sh libcuvs ${package_dir} cpp


# We temporarily override the values in `libcuvs` `pyproject.toml` so we can
# track the versions that have vendored NCCL (CUDA 11) vs. the ones that don't
# (CUDA 12+)
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
WHEEL_SIZE_LIMIT='0.9G'
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  WHEEL_SIZE_LIMIT='1.0G'
fi


rapids-logger "validate packages with 'pydistcheck'"
cd "${package_dir}"

pydistcheck \
    --inspect \
    --max-allowed-size-compressed $WHEEL_SIZE_LIMIT \
    "$(echo "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"/*.whl)"
