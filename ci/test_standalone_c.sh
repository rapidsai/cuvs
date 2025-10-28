#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail


# Download the standalone C library artifact
pkg_name="libcuvs_c_${RAPIDS_CUDA_VERSION}.tar.gz.zip"
rapids-logger "Download artifacts from previous jobs"
rapids-logger "pkg_name ${pkg_name}"
rapids-download-from-github "${pkg_name}"

# Extract the artifact to a staging directory
INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"
tar -xzf "${pkg_name}" -C "${INSTALL_PREFIX}"

rapids-logger "Run C API tests"

rapids-logger "C API tests completed successfully"
