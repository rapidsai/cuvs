#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key rust \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n rust

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate rust
set -eu

rapids-print-env

rapids-logger "Trying to find libclang"
find / -name libclang.so
export LIBCLANG_PATH=$(dirname $(find / -name libclang.so | head -n 1))
rapids-logger "LIBCLANG_PATH=$LIBCLANG_PATH"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# installing libcuvs/libraft will speed up the rust build substantially
rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libcuvs  \
  libraft

# build and test the rust bindings
cd rust
cargo test
