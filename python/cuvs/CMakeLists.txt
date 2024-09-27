# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

include(../../rapids_config.cmake)

# We always need CUDA for cuvs because the cuvs dependency brings in a header-only cuco dependency
# that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cuvs_py)

project(
  cuvs_py
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

# ##################################################################################################
# * User Options  --------------------------------------------------------------

option(FIND_CUVS_CPP "Search for existing CUVS C++ installations before defaulting to local files"
       OFF
)
option(USE_CUDA_MATH_WHEELS "Use the CUDA math wheels instead of the system libraries" OFF)

message(
  "CUVS_PY: Searching for existing cuVS C/C++ installations before defaulting to local files: ${FIND_CUVS_CPP}"
)

# ##################################################################################################
# * Process User Options  ------------------------------------------------------

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cython-core)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

# If the user requested it we attempt to find CUVS.
if(FIND_CUVS_CPP)
  find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS c_api)
  include(../../cpp/cmake/thirdparty/get_dlpack.cmake)
else()
  set(cuvs_FOUND OFF)
endif()

if(NOT cuvs_FOUND)
  find_package(CUDAToolkit REQUIRED)

  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY ON)

  # Statically link dependencies if building wheels
  set(CUDA_STATIC_RUNTIME ON)
  set(CUDA_STATIC_MATH_LIBRARIES ON)
  set(CUVS_USE_RAFT_STATIC ON)

  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
    set(CUDA_STATIC_MATH_LIBRARIES OFF)
  elseif(USE_CUDA_MATH_WHEELS)
    message(FATAL_ERROR "Cannot use CUDA math wheels with CUDA < 12.0")
  endif()

  add_subdirectory(../../cpp cuvs-cpp EXCLUDE_FROM_ALL)

  if(NOT CUDA_STATIC_MATH_LIBRARIES AND USE_CUDA_MATH_WHEELS)
    set(rpaths
        "$ORIGIN/../nvidia/cublas/lib"
        "$ORIGIN/../nvidia/curand/lib"
        "$ORIGIN/../nvidia/cusolver/lib"
        "$ORIGIN/../nvidia/cusparse/lib"
        "$ORIGIN/../nvidia/nvjitlink/lib"
    )
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
  endif()

  set(cython_lib_dir cuvs)
  install(TARGETS cuvs cuvs_c DESTINATION ${cython_lib_dir})
endif()

# ##################################################################################################
# * Build Cython artifacts -----------------------------------------------------

rapids_cython_init()

add_subdirectory(cuvs/common)
add_subdirectory(cuvs/distance)
add_subdirectory(cuvs/neighbors)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cuvs PATHS "${cython_lib_dir}")
endif()
