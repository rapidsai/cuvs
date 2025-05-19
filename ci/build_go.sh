#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key go \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n go

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate go
set -eu

rapids-print-env

export CGO_CFLAGS="-I${CONDA_PREFIX}/include"
export CGO_LDFLAGS="-L${CONDA_PREFIX}/lib -lcudart -lcuvs -lcuvs_c"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CC=clang

bash ./build.sh go
