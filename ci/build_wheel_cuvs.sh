#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Use RAFT wheel artifacts from PR
source ./ci/use_wheels_from_prs.sh

package_dir="python/cuvs"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcuvs wheels from this current build,
# then ensures 'cuvs' wheel builds always use the 'libcuvs' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# Using env variable PIP_CONSTRAINT (initialized by 'rapids-init-pip') is necessary to ensure the constraints
LIBCUVS_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUVS_WHEELHOUSE}"/libcuvs_*.whl)" >> "${PIP_CONSTRAINT}"

ci/build_wheel.sh cuvs ${package_dir}
ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
