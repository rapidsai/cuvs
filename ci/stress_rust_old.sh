#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CI-only: run the stress loop on the Rust crate as it was when #2233 was filed (before the
# DLPack/lifetime rewrite in #2245), against released libcuvs 26.08.
set -euo pipefail

OLD_SHA=025d44ceb7d07cc125e5f3bb4a7f0e9f0ca79148

. /opt/conda/etc/profile.d/conda.sh
conda config --set channel_priority flexible

rapids-dependency-file-generator \
  --output conda \
  --file-key rust \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  | sed -e 's/==26\.12\.\*,>=0\.0\.0a0/==26.08.*/' -e 's/==26\.12\.\*/==26.08.*/' \
        -e 's/^- rapidsai-nightly$/- rapidsai/' -e 's/^  - rapidsai-nightly$/  - rapidsai/' \
  | tee env_old.yaml

rapids-mamba-retry env create --yes -f env_old.yaml -n rustold
set +u
conda activate rustold
set -u
conda list | grep -E "libcuvs|libraft|librmm|cuda-version|^rust " || true

git fetch --depth 1 https://github.com/NVIDIA/cuvs.git "${OLD_SHA}"
mkdir -p old
git archive "${OLD_SHA}" rust | tar -x -C old
ls old/rust

ENV_NAME=rustold RUST_DIR=old/rust bash ci/stress_rust.sh
