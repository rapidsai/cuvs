#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# CI script to run clang-tidy with a full compilation database.
#
# This script:
# 1. Sets up a conda environment with build dependencies + clang tools
# 2. Runs cmake configure to generate compile_commands.json
# 3. Runs run-clang-tidy.py to check all C++ and CUDA files

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

# temporarily allow unbound variables for conda activation scripts
set +u
rapids-mamba-retry env create --yes -f env.yaml -n clang_tidy
conda activate clang_tidy
set -u

rapids-print-env

export CMAKE_GENERATOR=Ninja

rapids-logger "Configuring cmake (generating compilation database)"
cmake -S cpp -B cpp/build \
      -DCMAKE_CUDA_ARCHITECTURES=RAPIDS \
      -DBUILD_TESTS=ON \
      -DBUILD_C_LIBRARY=ON \
      -DBUILD_C_TESTS=ON \
      -DBUILD_MG_ALGOS=ON \
      -DBUILD_CUVS_BENCH=OFF

rapids-logger "Running clang-tidy"
python3 cpp/scripts/run-clang-tidy.py \
  -cdb cpp/build/compile_commands.json \
  -j "$(nproc)"
