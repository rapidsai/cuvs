#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-pip-retry install cmake
pyenv rehash

INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"

# Download the standalone C library artifact
if [ -z "$1" ]; then
  echo "Error: name of the standalone C library artifact is missing"
  exit 1
fi

payload_name="$1"
pkg_name="libcuvs_c.tar.gz"
rapids-logger "Download ${payload_name} artifacts from previous jobs"
DOWNLOAD_LOCATION=$(rapids-download-from-github "${payload_name}")

# Extract the artifact to a staging directory
tar -xf "${DOWNLOAD_LOCATION}/${pkg_name}" -C "${INSTALL_PREFIX}"


rapids-logger "Run C API tests"
ls -l "${INSTALL_PREFIX}"
cd "$INSTALL_PREFIX"/bin/gtests/libcuvs
ctest -j8 --output-on-failure

rapids-logger "C API tests completed successfully"
