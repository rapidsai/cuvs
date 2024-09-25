#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh


rapids-mamba-retry env create --yes -n go

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate go
set -eu

rapids-print-env

export CGO_CFLAGS="-I$CONDA_PREFIX/include -I/usr/local/cuda/include  -I/usr/local/include"
export CGO_LDFLAGS="-L$CONDA_PREFIX/lib -L/usr/local/cuda/lib64  -lcuvs_c -lcudart     -Wl,-rpath,$CONDA_PREFIX/lib-Wl,-rpath,/usr/local/cuda/lib64"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# installing libcuvs/libraft will speed up the rust build substantially
rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libcuvs  \
  libraft  \
  cuvs

bash ./build.sh go
