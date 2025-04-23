# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

include(rapids-cuda)
rapids_cuda_init_architectures(libcuvs-python)

project(
  libcuvs-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(USE_NCCL_RUNTIME_WHEEL "Use the NCCL wheel at runtime instead of the system library" OFF)

# Check if cuVS is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuvs package.
find_package(cuvs "${RAPIDS_VERSION}")

if(cuvs_FOUND)
  return()
endif()

unset(cuvs_FOUND)

# --- CUDA --- #
set(CUDA_STATIC_RUNTIME ON)
set(CUDA_STATIC_MATH_LIBRARIES OFF)

# --- RAFT ---#
set(CUVS_USE_RAFT_STATIC OFF)

# --- cuVS ---#
set(BUILD_TESTS OFF)
set(BUILD_C_LIBRARY ON)
set(CUVS_COMPILE_DYNAMIC_ONLY ON)

add_subdirectory(../../cpp cuvs-cpp)

# assumes libcuvs.so is installed 2 levels deep, e.g. site-packages/libcuvs/lib64/libcuvs.so
set(rpaths
    "$ORIGIN/../../nvidia/cublas/lib"
    "$ORIGIN/../../nvidia/curand/lib"
    "$ORIGIN/../../nvidia/cusolver/lib"
    "$ORIGIN/../../nvidia/cusparse/lib"
    "$ORIGIN/../../nvidia/nvjitlink/lib"
)

if(USE_NCCL_RUNTIME_WHEEL)
  list(APPEND rpaths "$ORIGIN/../../nvidia/nccl/lib")
endif()

set_property(
  TARGET cuvs
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
set_property(
  TARGET cuvs_c
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
