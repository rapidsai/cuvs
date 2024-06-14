#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key rust \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n rust

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate rust
set -eu

rapids-print-env

# we need to set up LIBCLANG_PATH to allow rust bindgen to work,
# grab it from the conda env
export LIBCLANG_PATH=$(dirname $(find /opt/conda -name libclang.so | head -n 1))
echo "LIBCLANG_PATH=$LIBCLANG_PATH"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# installing libcuvs/libraft will speed up the rust build substantially
rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libcuvs  \
  libraft

bash ./build.sh rust
