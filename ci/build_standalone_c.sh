#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

TOOLSET_VERSION=14
CMAKE_VERSION=3.31.8
CMAKE_ARCH=x86_64

BUILD_C_LIB_TESTS="OFF"
if [[ "${1:-}" == "--build-tests" ]]; then
  BUILD_C_LIB_TESTS="ON"
fi

dnf install -y \
      patch \
      tar \
      make

# Fetch and install CMake.
if [ ! -e "/usr/local/bin/cmake" ]; then
      pushd /usr/local
      wget --quiet https://github.com/Kitware/CMake/releases/download/v"${CMAKE_VERSION}"/cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}".tar.gz
      tar zxf cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}".tar.gz
      rm cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}".tar.gz
      ln -s /usr/local/cmake-"${CMAKE_VERSION}"-linux-"${CMAKE_ARCH}"/bin/cmake /usr/local/bin/cmake
      popd
fi

source rapids-configure-sccache
export SCCACHE_RECACHE=1

source rapids-date-string

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats


RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
export RAPIDS_ARTIFACTS_DIR

scl enable gcc-toolset-${TOOLSET_VERSION} -- \
      cmake -S cpp -B cpp/build/ \
            -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc \
            -DCMAKE_CUDA_ARCHITECTURES=RAPIDS \
            -DBUILD_SHARED_LIBS=OFF \
            -DCUTLASS_ENABLE_TESTS=OFF \
            -DDISABLE_OPENMP=OFF \
            -DBUILD_TESTS=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DCUVS_STATIC_RAPIDS_LIBRARIES=ON
cmake --build cpp/build "-j${PARALLEL_LEVEL}"

rapids-logger "Begin c build"

scl enable gcc-toolset-${TOOLSET_VERSION} -- \
      cmake -S c -B c/build \
            -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc \
            -DCUVSC_STATIC_CUVS_LIBRARY=ON \
            -DCMAKE_PREFIX_PATH="$PWD/cpp/build/" \
            -DBUILD_TESTS=${BUILD_C_LIB_TESTS}
cmake --build c/build "-j${PARALLEL_LEVEL}"

rapids-logger "Begin c install"
cmake --install c/build --prefix c/build/install

# need to install the tests
if [ "${BUILD_C_LIB_TESTS}" != "OFF" ]; then
      cmake --install c/build --prefix c/build/install --component testing
fi


rapids-logger "Begin gathering licenses"
cp LICENSE c/build/install/
if [ -e "./tool/extract_licenses_via_spdx.py" ]; then
      python ./tool/extract_licenses_via_spdx.py "." --with-licenses >> c/build/install/LICENSE
fi

rapids-logger "Begin c tarball creation"
tar czf libcuvs_c.tar.gz -C c/build/install/ .
ls -lh libcuvs_c.tar.gz

sccache --show-adv-stats
