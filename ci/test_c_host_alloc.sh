#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CI-only: run c/tests/core/host_alloc.cpp (pinned host buffer freed on another thread) against
# the nightly libcuvs (without the fix) and against this PR's libcuvs (with the fix).
set -uo pipefail

. /opt/conda/etc/profile.d/conda.sh
CUDA_MM=${RAPIDS_CUDA_VERSION%.*}
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuvs cuvs --cuda "$RAPIDS_CUDA_VERSION")")
nvidia-smi || true

build_and_run() {
  local label=$1
  local cxx
  cxx=$(ls "${CONDA_PREFIX}"/bin/*-linux-gnu-g++ | head -1)
  "${cxx}" -std=c++20 -O1 -g c/tests/core/host_alloc.cpp -o "host_alloc_${label}" \
    -I"${CONDA_PREFIX}/include" -I"${CONDA_PREFIX}/targets/x86_64-linux/include" \
    -L"${CONDA_PREFIX}/lib" -Wl,-rpath,"${CONDA_PREFIX}/lib" \
    -lcuvs_c -lgtest -lgtest_main -pthread || { echo "BUILD FAILED (${label})"; return; }
  local crashes=0 fails=0
  for i in $(seq 1 10); do
    "./host_alloc_${label}" > "run_${label}.log" 2>&1
    rc=$?
    [ "${i}" -eq 1 ] && tail -25 "run_${label}.log"
    if [ "${rc}" -ge 128 ]; then crashes=$((crashes + 1)); elif [ "${rc}" -ne 0 ]; then fails=$((fails + 1)); fi
    echo "${label} run ${i}: rc=${rc}"
  done
  echo "RESULT ${label}: 10 runs, ${crashes} crashes, ${fails} other failures"
}

PKGS=("cuda-version=${CUDA_MM}" cuda-cudart-dev dlpack gtest "gxx_linux-64=14.*" "sysroot_linux-64=2.28")

rapids-logger "Baseline: nightly libcuvs without the fix"
rapids-mamba-retry create -y -n cbase -c rapidsai-nightly -c conda-forge "libcuvs=26.12" "${PKGS[@]}"
set +u; conda activate cbase; set -u
conda list | grep -E "^libcuvs |^librmm "
build_and_run baseline
set +u; conda deactivate; set -u

rapids-logger "Fixed: this PR's libcuvs"
rapids-mamba-retry create -y -n cfix -c "${CPP_CHANNEL}" -c rapidsai-nightly -c conda-forge \
  "libcuvs=26.12" "libcuvs-tests=26.12" "${PKGS[@]}"
set +u; conda activate cfix; set -u
conda list | grep -E "^libcuvs |^libcuvs-tests |^librmm "
build_and_run fixed
GT="${CONDA_PREFIX}/bin/gtests/libcuvs/cuvs_c_host_alloc_test"
ls -la "${GT}" || true
"${GT}" --gtest_repeat=100 2>&1 | tail -15
echo "installed cuvs_c_host_alloc_test rc=${PIPESTATUS[0]}"
