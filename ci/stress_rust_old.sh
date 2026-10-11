#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CI-only: run the stress loop on the Rust crate as it was when #2233 was filed (before the
# DLPack/lifetime rewrite in #2245), against released libcuvs 26.06 (the reporter's setup).
set -euo pipefail

OLD_SHA=78135be4af3c7b458fd18b7b570c9b6978796daa

. /opt/conda/etc/profile.d/conda.sh
conda config --set channel_priority flexible

rapids-dependency-file-generator \
  --output conda \
  --file-key rust \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  | sed -e 's/==26\.12\.\*,>=0\.0\.0a0/==26.06.*/' -e 's/==26\.12\.\*/==26.06.*/' \
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
# The old crate is versioned 26.8.0 and requires a same-minor libcuvs; the reporter used 26.06.
sed -i 's/26\.8\.0/26.6.0/g' old/rust/Cargo.toml
grep -n version old/rust/Cargo.toml

ENV_NAME=rustold RUST_DIR=old/rust bash ci/stress_rust.sh
