#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate docs
set -eu

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libcuvs cuvs

export RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_NUMBER="$RAPIDS_VERSION_MAJOR_MINOR"
export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
popd

rapids-logger "Build Python docs"
pushd docs
sphinx-build -b dirhtml source _html
sphinx-build -b text source _text
mkdir -p "${RAPIDS_DOCS_DIR}/cuvs/"{html,txt}
mv _html/* "${RAPIDS_DOCS_DIR}/cuvs/html"
mv _text/* "${RAPIDS_DOCS_DIR}/cuvs/txt"
popd

rapids-upload-docs
