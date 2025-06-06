#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_MAJOR_MINOR

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs

# seeing failures on activating the environment here on unbound locals
# apply workaround from https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +eu
conda activate docs
set -eu

rapids-print-env

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
popd

rapids-logger "Build Rust docs"
pushd rust
LIBCLANG_PATH=$(dirname "$(find /opt/conda -name libclang.so | head -n 1)")
export LIBCLANG_PATH
cargo doc -p cuvs --no-deps
popd

rapids-logger "Build Python docs"
pushd docs
sphinx-build -b dirhtml source _html
mv ../rust/target/doc ./_html/_static/rust
mkdir -p "${RAPIDS_DOCS_DIR}/cuvs/"html
mv _html/* "${RAPIDS_DOCS_DIR}/cuvs/html"
popd

RAPIDS_VERSION_NUMBER="${RAPIDS_VERSION_MAJOR_MINOR}" rapids-upload-docs
