#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Sourced by ci/bench/build_<variant>.sh with BENCH_VARIANT set. Applies the variant's patch to
# this tree (upstream main), then builds libcuvs exactly like ci/build_cpp.sh and uploads it under
# a variant-specific artifact name.
set -euo pipefail
if [[ "${BENCH_VARIANT}" != "base" ]]; then
  git apply "ci/bench/patches/${BENCH_VARIANT}.patch"
fi
rapids-logger "Building libcuvs variant ${BENCH_VARIANT}"
git status --short
source ci/build_cpp.sh
RAPIDS_PACKAGE_NAME="${RAPIDS_PACKAGE_NAME}_${BENCH_VARIANT}"
export RAPIDS_PACKAGE_NAME
echo "Artifact name: ${RAPIDS_PACKAGE_NAME}"
