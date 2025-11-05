#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail


# Download the standalone C library artifact
payload_name="libcuvs_c_${RAPIDS_CUDA_VERSION}.tar.gz"
pkg_name="libcuvs_c.tar.gz"
rapids-logger "Download ${payload_name} artifacts from previous jobs"
DOWNLOAD_LOCATION=$(rapids-download-from-github "${payload_name}")

# Extract the artifact to a staging directory
INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"
ls -l "${DOWNLOAD_LOCATION}"
tar -xzf "${DOWNLOAD_LOCATION}/${pkg_name}" -C "${INSTALL_PREFIX}"

rapids-logger "Run C API tests"
cd "$INSTALL_PREFIX"/bin/gtests/libcuvs
ctest -j8 --output-on-failure

rapids-logger "C API tests completed successfully"
