#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

# Set up skbuild options. Enable sccache in skbuild config options
export SKBUILD_CONFIGURE_OPTIONS="-DDETECT_CONDA_ENV=OFF -DFIND_CUVS_CPP=OFF"

ci/build_wheel.sh cuvs python/cuvs
