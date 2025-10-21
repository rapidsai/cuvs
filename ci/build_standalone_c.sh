#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

TOOLSET_VERSION=14
CMAKE_VERSION=3.31.8
CMAKE_ARCH=x86_64

dnf install -y \
      gcc-toolset-${TOOLSET_VERSION} \
      patch \
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
      cmake -S cpp -B cpp/build/temp/ \
            -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc \
            -DCMAKE_CUDA_ARCHITECTURES=RAPIDS \
            -DBUILD_SHARED_LIBS=OFF \
            -DCUTLASS_ENABLE_TESTS=OFF \
            -DDISABLE_OPENMP=ON \
            -DBUILD_TESTS=OFF \
            -DBUILD_SHARED_LIBS=ON \
            -DCUVS_STATIC_RAPIDS_LIBRARIES=ON
cmake --build cpp/build/temp -j9

rapids-logger "Begin c build"

scl enable gcc-toolset-${TOOLSET_VERSION} -- \
      cmake -S c -B c/build/temp \
            -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-${TOOLSET_VERSION}/root/usr/bin/gcc \
            -DCUVSC_STATIC_CUVS_LIBRARY=ON \
            -DCMAKE_PREFIX_PATH="$PWD/cpp/build/temp" \
            -DBUILD_TESTS=OFF
cmake --build c/build/temp -j9

rapids-logger "Begin c install"
cmake --install c/build/temp --prefix c/build/temp/install

rapids-logger "Begin c tarball creation"
tar czf libcuvs_c.tar.gz -C c/build/temp/install/ .

sccache --show-adv-stats

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
