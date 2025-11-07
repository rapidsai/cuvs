#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

CMAKE_VERSION=3.31.8
CMAKE_ARCH=x86_64

# Fetch and install CMake.
if [ ! -e "/usr/local/bin/cmake" ]; then
      pushd /usr/local
      wget --quiet https://github.com/Kitware/CMake/releases/download/v"${CMAKE_VERSION}"/cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}".tar.gz
      tar zxf cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}".tar.gz
      rm cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}".tar.gz
      ln -s /usr/local/cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}"/bin/cmake /usr/local/bin/cmake
      popd
fi

# Download the standalone C library artifact
payload_name="libcuvs_c_${RAPIDS_CUDA_VERSION}.tar.gz"
pkg_name="libcuvs_c.tar.gz"
rapids-logger "Download ${payload_name} artifacts from previous jobs"
DOWNLOAD_LOCATION=$(rapids-download-from-github "${payload_name}")

# Extract the artifact to a staging directory
INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"
ls -l "${DOWNLOAD_LOCATION}"
tar -xf "${DOWNLOAD_LOCATION}/${pkg_name}" -C "${INSTALL_PREFIX}"

rapids-logger "Run C API tests"
ls -l "${INSTALL_PREFIX}"
cd "$INSTALL_PREFIX"/bin/gtests/libcuvs
ctest -j8 --output-on-failure

rapids-logger "C API tests completed successfully"
