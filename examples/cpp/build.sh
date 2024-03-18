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
set -e


if [[ ${CUVS_REPO_REL} != "" ]]; then
  CUVS_REPO_PATH="`readlink -f \"${CUVS_REPO_REL}\"`"
  EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DCPM_cuvs_SOURCE=${CUVS_REPO_PATH}"
fi

if [ "$1" == "clean" ]; then
  rm -rf build
  exit 0
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

cmake \
 -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
 -DCUVS_NVTX=OFF \
 -DCMAKE_CUDA_ARCHITECTURES="NATIVE" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 ${EXTRA_CMAKE_ARGS} \
 ../

cmake  --build . -j${PARALLEL_LEVEL}
