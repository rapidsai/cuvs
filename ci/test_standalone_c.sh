#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Download artifacts from previous jobs"

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"

# Download the standalone C library artifact
TARBALL="libcuvs_c.tar.gz"
rapids-download-from-github rapids-package-name "libcuvs_c"

# Extract the artifact to a staging directory
INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"
tar -xzf "${TARBALL}" -C "${INSTALL_PREFIX}"

rapids-logger "Run C API tests"

rapids-logger "C API tests completed successfully"
