#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

TOOLSET_VERSION=14
CMAKE_VERSION=3.31.8
CMAKE_ARCH=x86_64

dnf install -y \
      gcc-toolset-${TOOLSET_VERSION} \
      tar \
      make \
      wget

# Fetch and install CMake.
pushd /usr/local
wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
tar zxf cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
rm cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
ln -s /usr/local/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}/bin/cmake /usr/local/bin/cmake
popd

source rapids-configure-sccache

rapids-print-env

rapids-logger "Download artifacts from previous jobs"

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"

# Download the standalone C library artifact
TARBALL="libcuvs_c.tar.gz"
rapids-download-from-github "${TARBALL}"

# Extract the artifact to a staging directory
INSTALL_PREFIX="${PWD}/libcuvs_c_install"
mkdir -p "${INSTALL_PREFIX}"
tar -xzf "${TARBALL}" -C "${INSTALL_PREFIX}"

rapids-logger "Build C API tests"

sccache --zero-stats

# Build C tests against the standalone library
scl enable gcc-toolset-${TOOLSET_VERSION} -- \
      cmake -S c/tests -B c/tests/build \
            -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc \
            -DCMAKE_PREFIX_PATH="${INSTALL_PREFIX}" \
            -DBUILD_TESTS=ON

cmake --build c/tests/build -j9

rapids-logger "Run C API tests"

# Run the tests
cd c/tests/build
ctest --output-on-failure --no-tests=error

sccache --show-adv-stats

rapids-logger "C API tests completed successfully"

