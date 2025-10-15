# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

include(cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)
rapids_cpm_init()

# we want to use the already built libcuvs if its available, but the rust cmake-rs project doesn't
# support anything like find_package https://github.com/rust-lang/cmake-rs/issues/111 instead we're
# adding an extra level of indirection here - cmake-rs will attempt to build this project, and we'll
# using the existing libcuvs if its already built, and only fall back to building libcuvs if it
# isn't

project(
  cuvs-rs
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

option(FIND_CUVS_CPP "Search for existing CUVS C++ installations before defaulting to local files"
       ON
)

# If the user requested it we attempt to find CUVS.
if(FIND_CUVS_CPP)
  find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS c_api)
endif()

if(NOT cuvs_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY ON)
  add_subdirectory(../../cpp cuvs-cpp EXCLUDE_FROM_ALL)
endif()

include(get_dlpack.cmake)

# We are going to copy all the cuvs_c and dlpack headers into a staging location
if(TARGET cuvs_c)
  get_target_property(cuvs_c_headers cuvs_c INCLUDE_DIRECTORIES)
else()
  get_target_property(cuvs_c_headers cuvs::c_api INTERFACE_INCLUDE_DIRECTORIES)
endif()

file(
  INSTALL "${DLPACK_INCLUDE_DIR}"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bindings/include"
  FILES_MATCHING
  PATTERN "*.h"
)

foreach(cuvs_c_dir IN LISTS cuvs_c_headers)
  file(
    INSTALL "${cuvs_c_dir}"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bindings/include"
    FILES_MATCHING
    PATTERN "*.h"
  )
endforeach()

# add a dummy target here,
add_library(cuvs-rust INTERFACE)
target_link_libraries(cuvs-rust INTERFACE cuvs::cuvs)
install(TARGETS cuvs-rust)
