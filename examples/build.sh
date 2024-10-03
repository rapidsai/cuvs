#!/bin/bash

# Copyright (c) 2023-2024, NVIDIA CORPORATION.

# cuvs empty project template build script

# Abort script on first error
set -e


PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}

BUILD_TYPE=Release
BUILD_DIR=build/

CUVS_REPO_REL=""
EXTRA_CMAKE_ARGS=""
BUILD_ALL_GPU_ARCH=0
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi

# Root of examples
EXAMPLES_DIR=$(dirname "$(realpath "$0")")

if [[ ${CUVS_REPO_REL} != "" ]]; then
  CUVS_REPO_PATH="`readlink -f \"${CUVS_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_cuvs_SOURCE=${CUVS_REPO_PATH}"
else
  LIB_BUILD_DIR=${LIB_BUILD_DIR:-$(readlink -f "${EXAMPLES_DIR}/../cpp/build")}
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -Dcuvs_ROOT=${LIB_BUILD_DIR}"
fi

if [ "$1" == "clean" ]; then
  rm -rf build
  exit 0
fi

if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
    CUVS_CMAKE_CUDA_ARCHITECTURES="NATIVE"
    echo "Building for the architecture of the GPU in the system..."
else
    CUVS_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
    echo "Building for *ALL* supported GPU architectures..."
fi

################################################################################
# Add individual libcuvs examples build scripts down below

build_example() {
  example_dir=${1}
  example_dir="${EXAMPLES_DIR}/${example_dir}"
  build_dir="${example_dir}/build"

  # Configure
  cmake -S ${example_dir} -B ${build_dir} \
  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
  -DCUVS_NVTX=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=${CUVS_CMAKE_CUDA_ARCHITECTURES} \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  ${EXTRA_CMAKE_ARGS}
  # Build
  cmake --build ${build_dir} -j${PARALLEL_LEVEL}
}

build_example c
build_example cpp
