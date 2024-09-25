# =============================================================================
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
include(../rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

option(BUILD_CPU_ONLY "Build CPU only components. Applies to CUVS benchmarks currently" OFF)

# workaround for rapids_cuda_init_architectures not working for arch detection with
# enable_language(CUDA)
set(lang_list "CXX")

if(NOT BUILD_CPU_ONLY)
  include(rapids-cuda)
  rapids_cuda_init_architectures(CUVS)
  list(APPEND lang_list "CUDA")
endif()

project(
  CUVS
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${lang_list}
)
set(CMAKE_INSTALL_MESSAGE LAZY)

# Write the version header
rapids_cmake_write_version_file(include/cuvs/version_config.hpp)

# ##################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build cuvs shared libraries" ON)
option(BUILD_TESTS "Build cuvs unit-tests" ON)
option(BUILD_C_LIBRARY "Build cuVS C API library" OFF)
option(BUILD_C_TESTS "Build cuVS C API tests" OFF)
option(BUILD_ANN_BENCH "Build cuVS ann benchmarks" OFF)
option(BUILD_CAGRA_HNSWLIB "Build CAGRA+hnswlib interface" ON)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA runtime" OFF)
option(CUDA_STATIC_MATH_LIBRARIES "Statically link the CUDA math libraries" OFF)
option(CUVS_USE_RAFT_STATIC "Build and statically link the RAFT libraries" OFF)
option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(CUVS_NVTX "Enable nvtx markers" OFF)
option(CUVS_RAFT_CLONE_ON_PIN "Explicitly clone RAFT branch when pinned to non-feature branch" ON)

if((BUILD_TESTS OR BUILD_C_LIBRARY) AND NOT BUILD_CPU_ONLY)

endif()

if(BUILD_CPU_ONLY)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY OFF)
endif()

if(NOT BUILD_C_LIBRARY)
  set(BUILD_C_TESTS OFF)
endif()

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)
# cmake_dependent_option( CUVS_USE_FAISS_STATIC "Build and statically link the FAISS library for
# nearest neighbors search on GPU" ON CUVS_COMPILE_LIBRARY OFF )

message(VERBOSE "cuVS: Build cuVS unit-tests: ${BUILD_TESTS}")
message(VERBOSE "cuVS: Build CPU only components: ${BUILD_CPU_ONLY}")
message(VERBOSE "cuVS: Build ANN benchmarks: ${BUILD_ANN_BENCH}")
message(VERBOSE "cuVS: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "cuVS: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "cuVS: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "cuVS: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "cuVS: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "cuVS: Enable nvtx markers: ${CUVS_NVTX}")
message(VERBOSE "cuVS: Statically link the CUDA runtime: ${CUDA_STATIC_RUNTIME}")
message(VERBOSE "cuVS: Statically link the CUDA math libraries: ${CUDA_STATIC_MATH_LIBRARIES}")
message(VERBOSE "cuVS: Build and statically link RAFT libraries: ${CUVS_USE_RAFT_STATIC}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "cuVS: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# ##################################################################################################
# * Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(
      STATUS "cuVS: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}"
    )
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ##################################################################################################
# * compiler options ----------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_MATH_LIBRARIES)
  set(_ctk_static_suffix "_static")
endif()

if(NOT BUILD_CPU_ONLY)
  # CUDA runtime
  rapids_cuda_init_runtime(USE_STATIC ${CUDA_STATIC_RUNTIME})
  # * find CUDAToolkit package
  # * determine GPU architectures
  # * enable the CMake CUDA language
  # * set other CUDA compilation flags
  rapids_find_package(
    CUDAToolkit REQUIRED
    BUILD_EXPORT_SET cuvs-exports
    INSTALL_EXPORT_SET cuvs-exports
  )
else()
  add_compile_definitions(BUILD_CPU_ONLY)
endif()

if(NOT DISABLE_OPENMP)
  rapids_find_package(
    OpenMP REQUIRED
    BUILD_EXPORT_SET cuvs-exports
    INSTALL_EXPORT_SET cuvs-exports
  )
  if(OPENMP_FOUND)
    message(VERBOSE "cuVS: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

include(cmake/modules/ConfigureCUDA.cmake)

# ##################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

if(NOT BUILD_CPU_ONLY)
  include(cmake/thirdparty/get_raft.cmake)
endif()

if(BUILD_C_LIBRARY)
  include(cmake/thirdparty/get_dlpack.cmake)
endif()

if(BUILD_TESTS OR BUILD_C_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
endif()

include(cmake/thirdparty/get_cutlass.cmake)

if(BUILD_ANN_BENCH)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench(BUILD_STATIC)
endif()

if(BUILD_CAGRA_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib.cmake)
endif()

# ##################################################################################################
# * cuvs ---------------------------------------------------------------------
add_library(
  cuvs-cagra-search STATIC
  src/neighbors/cagra_search_float.cu
  src/neighbors/cagra_search_int8.cu
  src/neighbors/cagra_search_uint8.cu
  src/neighbors/detail/cagra/compute_distance.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_float_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_float_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_float_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_half_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_half_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_half_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_int8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_int8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_int8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_uint8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_uint8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_InnerProduct_uint8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_float_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_float_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_float_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_half_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_half_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_half_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_int8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_int8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_int8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_uint8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_uint8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/compute_distance_standard_L2Expanded_uint8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_float_uint32_dim128_t8_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_float_uint32_dim128_t8_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_float_uint32_dim256_t16_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_float_uint32_dim256_t16_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_float_uint32_dim512_t32_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_float_uint32_dim512_t32_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_half_uint32_dim128_t8_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_half_uint32_dim128_t8_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_half_uint32_dim256_t16_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_half_uint32_dim256_t16_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_half_uint32_dim512_t32_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_half_uint32_dim512_t32_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_int8_uint32_dim128_t8_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_int8_uint32_dim128_t8_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_int8_uint32_dim256_t16_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_int8_uint32_dim256_t16_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_int8_uint32_dim512_t32_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_int8_uint32_dim512_t32_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_uint8_uint32_dim128_t8_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_uint8_uint32_dim128_t8_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_uint8_uint32_dim256_t16_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_uint8_uint32_dim256_t16_8pq_4subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_uint8_uint32_dim512_t32_8pq_2subd_half.cu
  src/neighbors/detail/cagra/compute_distance_vpq_L2Expanded_uint8_uint32_dim512_t32_8pq_4subd_half.cu
  src/neighbors/detail/cagra/search_multi_cta_float_uint32.cu
  src/neighbors/detail/cagra/search_multi_cta_half_uint32.cu
  src/neighbors/detail/cagra/search_multi_cta_int8_uint32.cu
  src/neighbors/detail/cagra/search_multi_cta_uint8_uint32.cu
  src/neighbors/detail/cagra/search_multi_cta_float_uint64.cu
  src/neighbors/detail/cagra/search_multi_cta_half_uint64.cu
  src/neighbors/detail/cagra/search_single_cta_float_uint32.cu
  src/neighbors/detail/cagra/search_single_cta_half_uint32.cu
  src/neighbors/detail/cagra/search_single_cta_int8_uint32.cu
  src/neighbors/detail/cagra/search_single_cta_uint8_uint32.cu
  src/neighbors/detail/cagra/search_single_cta_float_uint64.cu
  src/neighbors/detail/cagra/search_single_cta_half_uint64.cu
)

file(GLOB_RECURSE compute_distance_sources "src/neighbors/detail/cagra/compute_distance_*.cu")
set_source_files_properties(${compute_distance_sources} PROPERTIES COMPILE_FLAGS -maxrregcount=64)

set_target_properties(
  cuvs-cagra-search
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             CUDA_SEPARABLE_COMPILATION ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(cuvs-cagra-search PRIVATE raft::raft)
target_include_directories(
  cuvs-cagra-search PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
target_compile_options(
  cuvs-cagra-search PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
)

add_library(
  cuvs SHARED
  src/cluster/kmeans_balanced_fit_float.cu
  src/cluster/kmeans_fit_float.cu
  src/cluster/kmeans_auto_find_k_float.cu
  src/cluster/kmeans_fit_predict_float.cu
  src/cluster/kmeans_predict_float.cu
  src/cluster/kmeans_balanced_fit_float.cu
  src/cluster/kmeans_balanced_fit_predict_float.cu
  src/cluster/kmeans_balanced_predict_float.cu
  src/cluster/kmeans_balanced_fit_int8.cu
  src/cluster/kmeans_balanced_fit_predict_int8.cu
  src/cluster/kmeans_balanced_predict_int8.cu
  src/cluster/kmeans_transform_float.cu
  src/cluster/single_linkage_float.cu
  src/distance/detail/pairwise_matrix/dispatch_canberra_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_canberra_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_canberra_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_correlation_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_correlation_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_correlation_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_cosine_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_cosine_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_cosine_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hamming_unexpanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hamming_unexpanded_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hamming_unexpanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hellinger_expanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hellinger_expanded_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hellinger_expanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_jensen_shannon_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_jensen_shannon_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_jensen_shannon_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_kl_divergence_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_kl_divergence_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_kl_divergence_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l1_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l1_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l1_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_expanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_expanded_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_expanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_unexpanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_unexpanded_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_unexpanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l_inf_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l_inf_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l_inf_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_lp_unexpanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_lp_unexpanded_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_lp_unexpanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_russel_rao_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_russel_rao_half_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_russel_rao_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_rbf.cu
  src/distance/detail/fused_distance_nn.cu
  src/distance/distance.cu
  src/distance/pairwise_distance.cu
  src/neighbors/brute_force.cu
  src/neighbors/cagra_build_float.cu
  src/neighbors/cagra_build_int8.cu
  src/neighbors/cagra_build_uint8.cu
  src/neighbors/cagra_extend_float.cu
  src/neighbors/cagra_extend_int8.cu
  src/neighbors/cagra_extend_uint8.cu
  src/neighbors/cagra_optimize.cu
  src/neighbors/cagra_serialize_float.cu
  src/neighbors/cagra_serialize_int8.cu
  src/neighbors/cagra_serialize_uint8.cu
  src/neighbors/detail/cagra/cagra_build.cpp
  src/neighbors/detail/cagra/topk_for_cagra/topk.cu
  $<$<BOOL:${BUILD_CAGRA_HNSWLIB}>:src/neighbors/hnsw.cpp>
  src/neighbors/ivf_flat_index.cpp
  src/neighbors/ivf_flat/ivf_flat_build_extend_float_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_build_extend_int8_t_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_build_extend_uint8_t_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_helpers.cu
  src/neighbors/ivf_flat/ivf_flat_search_float_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_search_int8_t_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_search_uint8_t_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_serialize_float_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_serialize_int8_t_int64_t.cu
  src/neighbors/ivf_flat/ivf_flat_serialize_uint8_t_int64_t.cu
  src/neighbors/ivf_pq_index.cpp
  src/neighbors/ivf_pq/ivf_pq_build_common.cu
  src/neighbors/ivf_pq/ivf_pq_serialize.cu
  src/neighbors/ivf_pq/ivf_pq_deserialize.cu
  src/neighbors/ivf_pq/detail/ivf_pq_build_extend_float_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_build_extend_int8_t_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_build_extend_uint8_t_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_half_fp8_false.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_half_fp8_true.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_half_half.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_half.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_float.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_fp8_false.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_fp8_true.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_half_fp8_false_bitset64.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_half_fp8_true_bitset64.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_half_half_bitset64.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_half_bitset64.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_float_bitset64.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_fp8_false_bitset64.cu
  src/neighbors/ivf_pq/detail/ivf_pq_compute_similarity_float_fp8_true_bitset64.cu
  src/neighbors/ivf_pq/detail/ivf_pq_search_float_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_search_int8_t_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_search_uint8_t_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_search_with_filter_float_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_search_with_filter_int8_t_int64_t.cu
  src/neighbors/ivf_pq/detail/ivf_pq_search_with_filter_uint8_t_int64_t.cu
  src/neighbors/nn_descent.cu
  src/neighbors/nn_descent_float.cu
  src/neighbors/nn_descent_int8.cu
  src/neighbors/nn_descent_uint8.cu
  src/neighbors/refine/detail/refine_device_float_float.cu
  src/neighbors/refine/detail/refine_device_half_float.cu
  src/neighbors/refine/detail/refine_device_int8_t_float.cu
  src/neighbors/refine/detail/refine_device_uint8_t_float.cu
  src/neighbors/refine/detail/refine_host_float_float.cpp
  src/neighbors/refine/detail/refine_host_half_float.cpp
  src/neighbors/refine/detail/refine_host_int8_t_float.cpp
  src/neighbors/refine/detail/refine_host_uint8_t_float.cpp
  src/neighbors/sample_filter.cu
  src/selection/select_k_float_int64_t.cu
  src/selection/select_k_float_uint32_t.cu
  src/selection/select_k_half_uint32_t.cu
  src/stats/silhouette_score.cu
  src/stats/trustworthiness_score.cu
)

target_compile_definitions(cuvs PRIVATE "CUVS_EXPLICIT_INSTANTIATE_ONLY")

target_compile_options(
  cuvs INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda
                 --expt-relaxed-constexpr>
)

add_library(cuvs::cuvs ALIAS cuvs)

target_include_directories(
  cuvs PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
              "$<INSTALL_INTERFACE:include>"
)

rapids_find_package(
  OpenMP REQUIRED
  BUILD_EXPORT_SET cuvs-exports
  INSTALL_EXPORT_SET cuvs-exports
)

if(NOT BUILD_CPU_ONLY)

  set(CUVS_CUSOLVER_DEPENDENCY CUDA::cusolver${_ctk_static_suffix})
  set(CUVS_CUBLAS_DEPENDENCY CUDA::cublas${_ctk_static_suffix})
  set(CUVS_CURAND_DEPENDENCY CUDA::curand${_ctk_static_suffix})
  set(CUVS_CUSPARSE_DEPENDENCY CUDA::cusparse${_ctk_static_suffix})

  set(CUVS_CTK_MATH_DEPENDENCIES ${CUVS_CUBLAS_DEPENDENCY} ${CUVS_CUSOLVER_DEPENDENCY}
                                 ${CUVS_CUSPARSE_DEPENDENCY} ${CUVS_CURAND_DEPENDENCY}
  )

  # Keep cuVS as lightweight as possible. Only CUDA libs and rmm should be used in global target.
  target_link_libraries(
    cuvs
    PUBLIC rmm::rmm raft::raft ${CUVS_CTK_MATH_DEPENDENCIES}
    PRIVATE nvidia::cutlass::cutlass $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX> cuvs-cagra-search
  )
endif()

if(BUILD_CAGRA_HNSWLIB)
  target_link_libraries(cuvs PRIVATE hnswlib::hnswlib)
  target_compile_definitions(cuvs PUBLIC CUVS_BUILD_CAGRA_HNSWLIB)
endif()

# Endian detection
include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
  target_compile_definitions(cuvs PRIVATE CUVS_SYSTEM_LITTLE_ENDIAN=0)
else()
  target_compile_definitions(cuvs PRIVATE CUVS_SYSTEM_LITTLE_ENDIAN=1)
endif()

file(
  WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
  [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
)

# ##################################################################################################
# * NVTX support in cuvs -----------------------------------------------------

if(CUVS_NVTX)
  # This enables NVTX within the project with no option to disable it downstream.
  target_link_libraries(cuvs PUBLIC CUDA::nvtx3)
  target_compile_definitions(cuvs PUBLIC NVTX_ENABLED)
else()
  # Allow enable NVTX downstream if not set here. This creates a new option at build/install time,
  # which is set by default to OFF, but can be enabled in the dependent project.
  get_property(
    nvtx_option_help_string
    CACHE CUVS_NVTX
    PROPERTY HELPSTRING
  )
  string(
    CONCAT
      nvtx_export_string
      "option(CUVS_NVTX \""
      ${nvtx_option_help_string}
      "\" OFF)"
      [=[

target_link_libraries(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:CUDA::nvtx3>)
target_compile_definitions(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>)

  ]=]
  )
endif()

set_target_properties(
  cuvs
  PROPERTIES BUILD_RPATH "\$ORIGIN"
             INSTALL_RPATH "\$ORIGIN"
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cuvs PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
               "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
)
# ensure CUDA symbols aren't relocated to the middle of the debug build binaries
target_link_options(cuvs PRIVATE $<HOST_LINK:${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld>)

# ##################################################################################################
# * cuvs_c -------------------------------------------------------------------------------
if(BUILD_C_LIBRARY)
  add_library(
    cuvs_c SHARED
    src/core/c_api.cpp
    src/neighbors/brute_force_c.cpp
    src/neighbors/ivf_flat_c.cpp
    src/neighbors/ivf_pq_c.cpp
    src/neighbors/cagra_c.cpp
    src/neighbors/hnsw_c.cpp
    src/neighbors/refine/refine_c.cpp
    src/distance/pairwise_distance_c.cpp
  )

  add_library(cuvs::c_api ALIAS cuvs_c)

  set_target_properties(
    cuvs_c
    PROPERTIES BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               EXPORT_NAME c_api
  )

  target_compile_options(cuvs_c PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>")

  target_include_directories(
    cuvs_c
    PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
    INTERFACE "$<INSTALL_INTERFACE:include>"
  )

  target_link_libraries(
    cuvs_c
    PUBLIC cuvs::cuvs ${CUVS_CTK_MATH_DEPENDENCIES}
    PRIVATE raft::raft
  )

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(cuvs_c PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

# ##################################################################################################
# * install targets-----------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(GNUInstallDirs)
include(CPack)

install(
  TARGETS cuvs
  DESTINATION ${lib_dir}
  COMPONENT cuvs
  EXPORT cuvs-exports
)

install(
  DIRECTORY include/cuvs
  COMPONENT cuvs
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUILD_C_LIBRARY)
  install(
    TARGETS cuvs_c
    DESTINATION ${lib_dir}
    COMPONENT c_api
    EXPORT cuvs-c-exports
  )
endif()

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuvs/version_config.hpp
  COMPONENT cuvs
  DESTINATION include/cuvs
)

if(TARGET cuvs_c)
  list(APPEND cuvs_components c_api)
  list(APPEND cuvs_export_sets cuvs-c-exports)
  set(CUVS_C_TARGET cuvs_c)
endif()

# Use `rapids_export` for 22.04 as it will have COMPONENT support
rapids_export(
  INSTALL cuvs
  EXPORT_SET cuvs-exports
  COMPONENTS ${cuvs_components}
  COMPONENTS_EXPORT_SET ${cuvs_export_sets}
  GLOBAL_TARGETS cuvs ${CUVS_C_TARGET}
  NAMESPACE cuvs::
)

# ##################################################################################################
# * build export -------------------------------------------------------------
rapids_export(
  BUILD cuvs
  EXPORT_SET cuvs-exports
  COMPONENTS ${cuvs_components}
  COMPONENTS_EXPORT_SET ${cuvs_export_sets}
  GLOBAL_TARGETS cuvs ${CUVS_C_TARGET}
  NAMESPACE cuvs::
)

# ##################################################################################################
# * build test executable ----------------------------------------------------

if(BUILD_TESTS OR BUILD_C_TESTS)
  add_subdirectory(internal)
  add_subdirectory(test)
endif()

# ##################################################################################################
# * build ann benchmark executable -----------------------------------------------

if(BUILD_ANN_BENCH)
  add_subdirectory(bench/ann/)
endif()
