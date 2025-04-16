#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cuvs"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Downloads libcuvs wheels from this current build,
# then ensures 'cuvs' wheel builds always use the 'libcuvs' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcuvs_dist
echo "libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcuvs_dist/libcuvs_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

ci/build_wheel.sh cuvs ${package_dir} python
ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
