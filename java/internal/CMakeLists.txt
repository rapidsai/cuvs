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
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)
rapids_cpm_init()

project(
  cuvs-java
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX C
)

find_package(OpenMP)
find_package(Threads)

option(FIND_CUVS_CPP "Search for existing CUVS C++ installations before defaulting to local files"
       ON
)

if(FIND_CUVS_CPP)
  find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS c_api)
  if(NOT TARGET cuvs::c_api)
    message(
      FATAL_ERROR
        "Building against a preexisting libcuvs library requires the compiled libcuvs to have been built!"
    )
  endif()
else()
  set(cuvs_FOUND OFF)
endif()

if(NOT cuvs_FOUND)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY ON)
  add_subdirectory(../../cpp cuvs-cpp EXCLUDE_FROM_ALL)
endif()

include(get_dlpack.cmake)
