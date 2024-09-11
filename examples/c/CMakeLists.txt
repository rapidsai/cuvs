# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../cmake/thirdparty/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

project(test_cuvs_c LANGUAGES C CXX CUDA)

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY ON)
include(../cmake/thirdparty/get_dlpack.cmake)
include(../cmake/thirdparty/get_cuvs.cmake)

add_executable(CAGRA_C_EXAMPLE src/cagra_c_example.c)
target_include_directories(CAGRA_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(CAGRA_C_EXAMPLE PRIVATE cuvs::c_api $<TARGET_NAME_IF_EXISTS:conda_env>)

add_executable(L2_C_EXAMPLE src/L2_c_example.c)
target_include_directories(L2_C_EXAMPLE PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")
target_link_libraries(L2_C_EXAMPLE PRIVATE cuvs::c_api $<TARGET_NAME_IF_EXISTS:conda_env>)
