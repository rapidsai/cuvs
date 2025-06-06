#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key rust \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n rust

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate rust
set -eu

rapids-print-env

# we need to set up LIBCLANG_PATH to allow rust bindgen to work,
# grab it from the conda env
LIBCLANG_PATH=$(dirname "$(find /opt/conda -name libclang.so | head -n 1)")
export LIBCLANG_PATH
echo "LIBCLANG_PATH=$LIBCLANG_PATH"

bash ./build.sh rust

# Also test out that we can publish cuvs-sys via a dry-run
pushd ./rust/cuvs-sys
cargo publish --dry-run
popd
