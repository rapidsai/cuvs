# =============================================================================
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

include(../../fetch_rapids.cmake)

set(cuvs_version 24.02.00)

# We always need CUDA for cuvs because the cuvs dependency brings in a header-only cuco
# dependency that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cuvs)

project(
  cuvs
  VERSION ${cuvs_version}
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

option(FIND_CUVS_CPP "Search for existing CUVS C++ installations before defaulting to local files"
       ON
)

option(CUVS_BUILD_WHEELS "Whether this build is generating a Python wheel." OFF)

# If the user requested it we attempt to find CUVS.
if(FIND_CUVS_CPP)
  find_package(cuvs ${cuvs_version} REQUIRED COMPONENTS compiled)
  if(NOT TARGET cuvs::cuvs_lib)
    message(
      FATAL_ERROR
        "Building against a preexisting libcuvs library requires the compiled libcuvs to have been built!"
    )

  endif()
else()
  set(cuvs_FOUND OFF)
endif()

include(rapids-cython)

if(NOT cuvs_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_PRIMS_BENCH OFF)
  set(BUILD_ANN_BENCH OFF)
  set(CUVS_COMPILE_LIBRARY ON)

  set(_exclude_from_all "")
  if(CUVS_BUILD_WHEELS)
    # Statically link dependencies if building wheels
    set(CUDA_STATIC_RUNTIME ON)
    # Don't install the cuvs C++ targets into wheels
    set(_exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_subdirectory(../../cpp cuvs-cpp ${_exclude_from_all})

  # When building the C++ libraries from source we must copy libcuvs.so alongside the
  # pairwise_distance and random Cython libraries TODO: when we have a single 'compiled' cuvs
  # library, we shouldn't need this
  set(cython_lib_dir cuvs)
  install(TARGETS cuvs_lib DESTINATION ${cython_lib_dir})
endif()

rapids_cython_init()

add_subdirectory(cuvs/common)
add_subdirectory(cuvs/distance)
add_subdirectory(cuvs/matrix)
add_subdirectory(cuvs/neighbors)
add_subdirectory(cuvs/random)
add_subdirectory(cuvs/cluster)

if(DEFINED cython_lib_dir)
  rapids_cython_add_rpath_entries(TARGET cuvs PATHS "${cython_lib_dir}")
endif()
