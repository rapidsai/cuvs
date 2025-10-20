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
      wget

source scl_source enable gcc-toolset-${TOOLSET_VERSION}

# Fetch and install CMake.
pushd /usr/local
wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
tar zxf cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
rm cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz
ln -s /usr/local/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}/bin/cmake /usr/local/bin/cmake
popd

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats


RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
export RAPIDS_ARTIFACTS_DIR


cmake -S cpp -B cpp/build/ \
      -DCMAKE_CUDA_ARCHITECTURES=RAPIDS \
      -DBUILD_SHARED_LIBS=OFF \
      -DCUTLASS_ENABLE_TESTS=OFF \
      -DDISABLE_OPENMP=ON \
      -DBUILD_TESTS=OFF \
      -DBUILD_SHARED_LIBS=ON \
      -DCUVS_STATIC_RAPIDS_LIBRARIES=ON
cmake --build cpp/build


rapids-logger "Begin c build"

cmake -S c -B c/build \
      -DCUVSC_STATIC_CUVS_LIBRARY=ON \
      -DCMAKE_PREFIX_PATH="cpp/build" \
      -DBUILD_TESTS=OFF && \
cmake --install c/build --prefix c/build/install
tar czf libcuvs_c.tar.gz -C c/build/install/ .

sccache --show-adv-stats

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
