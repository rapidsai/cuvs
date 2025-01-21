#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cuvs"

case "${RAPIDS_CUDA_VERSION}" in
  12.*)
    EXTRA_CMAKE_ARGS="-DUSE_CUDA_MATH_WHEELS=ON"
  ;;
  11.*)
    EXTRA_CMAKE_ARGS="-DUSE_CUDA_MATH_WHEELS=OFF"
  ;;
esac

# Set up skbuild options. Enable sccache in skbuild config options
export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;${EXTRA_CMAKE_ARGS}"

ci/build_wheel.sh cuvs ${package_dir}
ci/validate_wheel.sh ${package_dir} final_dist
