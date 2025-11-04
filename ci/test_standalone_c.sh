#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail


# Download the standalone C library artifact
pkg_key="libcuvs_c_${RAPIDS_CUDA_VERSION}.tar.gz"
pkg_name="libcuvs_c_${RAPIDS_CUDA_VERSION}.tar.gz"
rapids-logger "Download artifacts from previous jobs"
rapids-logger "pkg_key ${pkg_key}"
rapids-download-from-github "${pkg_key}"

# Extract the artifact to a staging directory
INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"
unzip "${pkg_name}.zip"
tar -xzf "${pkg_name}" -C "${INSTALL_PREFIX}"

rapids-logger "Run C API tests"

rapids-logger "C API tests completed successfully"
