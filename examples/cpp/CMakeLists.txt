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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# ------------- configure rapids-cmake --------------#

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# ------------- configure project --------------#

rapids_cuda_init_architectures(test_cuvs)

project(test_cuvs LANGUAGES CXX CUDA)
find_package(Threads)

# ------------- configure cuvs -----------------#

rapids_cpm_init()
set(BUILD_CUVS_C_LIBRARY OFF)
include(../cmake/thirdparty/get_cuvs.cmake)

# -------------- compile tasks ----------------- #
add_executable(BRUTE_FORCE_EXAMPLE src/brute_force_bitmap.cu)
add_executable(CAGRA_EXAMPLE src/cagra_example.cu)
add_executable(CAGRA_PERSISTENT_EXAMPLE src/cagra_persistent_example.cu)
add_executable(DYNAMIC_BATCHING_EXAMPLE src/dynamic_batching_example.cu)
add_executable(IVF_FLAT_EXAMPLE src/ivf_flat_example.cu)
add_executable(IVF_PQ_EXAMPLE src/ivf_pq_example.cu)
add_executable(VAMANA_EXAMPLE src/vamana_example.cu)

# `$<TARGET_NAME_IF_EXISTS:conda_env>` is a generator expression that ensures that targets are
# installed in a conda environment, if one exists
target_link_libraries(BRUTE_FORCE_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(CAGRA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(
  CAGRA_PERSISTENT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env> Threads::Threads
)
target_link_libraries(
  DYNAMIC_BATCHING_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env> Threads::Threads
)
target_link_libraries(IVF_PQ_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(IVF_FLAT_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
target_link_libraries(VAMANA_EXAMPLE PRIVATE cuvs::cuvs $<TARGET_NAME_IF_EXISTS:conda_env>)
