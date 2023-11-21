# =============================================================================
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
set(RAPIDS_VERSION "24.02")
set(CUVS_VERSION "24.02.00")

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
include(../fetch_rapids.cmake)
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
  rapids_cuda_init_architectures(cuVS)
  list(APPEND lang_list "CUDA")
endif()

project(
  cuVS
  VERSION ${CUVS_VERSION}
  LANGUAGES ${lang_list}
)

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
option(BUILD_MICRO_BENCH "Build cuvs C++ micro benchmarks" OFF)
option(BUILD_ANN_BENCH "Build cuvs ann benchmarks" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)
option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(CUVS_NVTX "Enable nvtx markers" OFF)

if((BUILD_TESTS
    OR BUILD_MICRO_BENCH
    OR BUILD_ANN_BENCH
   )
   AND NOT BUILD_CPU_ONLY
)

endif()

if(BUILD_CPU_ONLY)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTS OFF)
endif()

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)
# cmake_dependent_option( CUVS_USE_FAISS_STATIC "Build and statically link the FAISS library for
# nearest neighbors search on GPU" ON CUVS_COMPILE_LIBRARY OFF )

message(VERBOSE "cuVS: Build cuVS unit-tests: ${BUILD_TESTS}")
message(VERBOSE "cuVS: Building cuvs C++ benchmarks: ${BUILD_MICRO_BENCH}")
message(VERBOSE "cuVS: Building ANN benchmarks: ${BUILD_ANN_BENCH}")
message(VERBOSE "cuVS: Build CPU only components: ${BUILD_CPU_ONLY}")
message(VERBOSE "cuVS: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "cuVS: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "cuVS: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "cuVS: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "cuVS: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "cuVS: Enable nvtx markers: ${CUVS_NVTX}")
message(VERBOSE
        "cuVS: Statically link the CUDA toolkit runtime and libraries: ${CUDA_STATIC_RUNTIME}"
)

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
if(CUDA_STATIC_RUNTIME)
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

if(BUILD_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

if(BUILD_MICRO_BENCH OR BUILD_ANN_BENCH)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench()
endif()

# ##################################################################################################
# * cuvs ---------------------------------------------------------------------

add_library(
  cuvs SHARED
  src/distance/detail/pairwise_matrix/dispatch_canberra_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_canberra_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_correlation_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_correlation_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_cosine_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_cosine_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hamming_unexpanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hamming_unexpanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hellinger_expanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_hellinger_expanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_jensen_shannon_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_jensen_shannon_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_kl_divergence_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_kl_divergence_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l1_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l1_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_expanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_expanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_unexpanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l2_unexpanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l_inf_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_l_inf_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_lp_unexpanded_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_lp_unexpanded_float_float_float_int.cu
  src/distance/detail/pairwise_matrix/dispatch_rbf.cu
  src/distance/detail/pairwise_matrix/dispatch_russel_rao_double_double_double_int.cu
  src/distance/detail/pairwise_matrix/dispatch_russel_rao_float_float_float_int.cu
  src/distance/distance.cu
  src/distance/fused_l2_nn.cu
  src/matrix/detail/select_k_double_int64_t.cu
  src/matrix/detail/select_k_double_uint32_t.cu
  src/matrix/detail/select_k_float_int64_t.cu
  src/matrix/detail/select_k_float_uint32_t.cu
  src/matrix/detail/select_k_float_int32.cu
  src/matrix/detail/select_k_half_int64_t.cu
  src/matrix/detail/select_k_half_uint32_t.cu
  src/neighbors/ball_cover.cu
  src/neighbors/brute_force_fused_l2_knn_float_int64_t.cu
  src/neighbors/brute_force_knn_int64_t_float_int64_t.cu
  src/neighbors/brute_force_knn_int64_t_float_uint32_t.cu
  src/neighbors/brute_force_knn_int_float_int.cu
  src/neighbors/brute_force_knn_uint32_t_float_uint32_t.cu
  src/neighbors/brute_force_knn_index_float.cu
  src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/search_multi_cta_float_uint32_dim1024_t32.cu
  src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/search_multi_cta_int8_uint32_dim1024_t32.cu
  src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/search_multi_cta_uint8_uint32_dim1024_t32.cu
  src/neighbors/detail/cagra/search_single_cta_float_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/search_single_cta_float_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/search_single_cta_float_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/search_single_cta_float_uint32_dim1024_t32.cu
  src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/search_single_cta_int8_uint32_dim1024_t32.cu
  src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim128_t8.cu
  src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim256_t16.cu
  src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim512_t32.cu
  src/neighbors/detail/cagra/search_single_cta_uint8_uint32_dim1024_t32.cu
  src/neighbors/detail/ivf_flat_interleaved_scan_float_float_int64_t.cu
  src/neighbors/detail/ivf_flat_interleaved_scan_int8_t_int32_t_int64_t.cu
  src/neighbors/detail/ivf_flat_interleaved_scan_uint8_t_uint32_t_int64_t.cu
  src/neighbors/detail/ivf_flat_search.cu
  src/neighbors/detail/ivf_pq_compute_similarity_float_float.cu
  src/neighbors/detail/ivf_pq_compute_similarity_float_fp8_false.cu
  src/neighbors/detail/ivf_pq_compute_similarity_float_fp8_true.cu
  src/neighbors/detail/ivf_pq_compute_similarity_float_half.cu
  src/neighbors/detail/ivf_pq_compute_similarity_half_fp8_false.cu
  src/neighbors/detail/ivf_pq_compute_similarity_half_fp8_true.cu
  src/neighbors/detail/ivf_pq_compute_similarity_half_half.cu
  src/neighbors/detail/refine_host_float_float.cpp
  src/neighbors/detail/refine_host_int8_t_float.cpp
  src/neighbors/detail/refine_host_uint8_t_float.cpp
  src/neighbors/detail/selection_faiss_int32_t_float.cu
  src/neighbors/detail/selection_faiss_int_double.cu
  src/neighbors/detail/selection_faiss_long_float.cu
  src/neighbors/detail/selection_faiss_size_t_double.cu
  src/neighbors/detail/selection_faiss_size_t_float.cu
  src/neighbors/detail/selection_faiss_uint32_t_float.cu
  src/neighbors/detail/selection_faiss_int64_t_double.cu
  src/neighbors/detail/selection_faiss_int64_t_half.cu
  src/neighbors/detail/selection_faiss_uint32_t_double.cu
  src/neighbors/detail/selection_faiss_uint32_t_half.cu
  src/neighbors/ivf_flat_build_float_int64_t.cu
  src/neighbors/ivf_flat_build_int8_t_int64_t.cu
  src/neighbors/ivf_flat_build_uint8_t_int64_t.cu
  src/neighbors/ivf_flat_extend_float_int64_t.cu
  src/neighbors/ivf_flat_extend_int8_t_int64_t.cu
  src/neighbors/ivf_flat_extend_uint8_t_int64_t.cu
  src/neighbors/ivf_flat_search_float_int64_t.cu
  src/neighbors/ivf_flat_search_int8_t_int64_t.cu
  src/neighbors/ivf_flat_search_uint8_t_int64_t.cu
  src/neighbors/ivfpq_build_float_int64_t.cu
  src/neighbors/ivfpq_build_int8_t_int64_t.cu
  src/neighbors/ivfpq_build_uint8_t_int64_t.cu
  src/neighbors/ivfpq_extend_float_int64_t.cu
  src/neighbors/ivfpq_extend_int8_t_int64_t.cu
  src/neighbors/ivfpq_extend_uint8_t_int64_t.cu
  src/neighbors/ivfpq_search_float_int64_t.cu
  src/neighbors/ivfpq_search_int8_t_int64_t.cu
  src/neighbors/ivfpq_search_uint8_t_int64_t.cu
  src/neighbors/refine_float_float.cu
  src/neighbors/refine_int8_t_float.cu
  src/neighbors/refine_uint8_t_float.cu
  src/cuvs_runtime/cluster/cluster_cost.cuh
  src/cuvs_runtime/cluster/cluster_cost_double.cu
  src/cuvs_runtime/cluster/cluster_cost_float.cu
  src/cuvs_runtime/cluster/kmeans_fit_double.cu
  src/cuvs_runtime/cluster/kmeans_fit_float.cu
  src/cuvs_runtime/cluster/kmeans_init_plus_plus_double.cu
  src/cuvs_runtime/cluster/kmeans_init_plus_plus_float.cu
  src/cuvs_runtime/cluster/update_centroids.cuh
  src/cuvs_runtime/cluster/update_centroids_double.cu
  src/cuvs_runtime/cluster/update_centroids_float.cu
  src/cuvs_runtime/distance/fused_l2_min_arg.cu
  src/cuvs_runtime/distance/pairwise_distance.cu
  src/cuvs_runtime/matrix/select_k_float_int64_t.cu
  src/cuvs_runtime/neighbors/brute_force_knn_int64_t_float.cu
  src/cuvs_runtime/neighbors/cagra_build.cu
  src/cuvs_runtime/neighbors/cagra_search.cu
  src/cuvs_runtime/neighbors/cagra_serialize.cu
  src/cuvs_runtime/neighbors/ivf_flat_build.cu
  src/cuvs_runtime/neighbors/ivf_flat_search.cu
  src/cuvs_runtime/neighbors/ivf_flat_serialize.cu
  src/cuvs_runtime/neighbors/ivfpq_build.cu
  src/cuvs_runtime/neighbors/ivfpq_deserialize.cu
  src/cuvs_runtime/neighbors/ivfpq_search_float_int64_t.cu
  src/cuvs_runtime/neighbors/ivfpq_search_int8_t_int64_t.cu
  src/cuvs_runtime/neighbors/ivfpq_search_uint8_t_int64_t.cu
  src/cuvs_runtime/neighbors/ivfpq_serialize.cu
  src/cuvs_runtime/neighbors/refine_d_int64_t_float.cu
  src/cuvs_runtime/neighbors/refine_d_int64_t_int8_t.cu
  src/cuvs_runtime/neighbors/refine_d_int64_t_uint8_t.cu
  src/cuvs_runtime/neighbors/refine_h_int64_t_float.cu
  src/cuvs_runtime/neighbors/refine_h_int64_t_int8_t.cu
  src/cuvs_runtime/neighbors/refine_h_int64_t_uint8_t.cu
  src/cuvs_runtime/random/rmat_rectangular_generator_int64_double.cu
  src/cuvs_runtime/random/rmat_rectangular_generator_int64_float.cu
  src/cuvs_runtime/random/rmat_rectangular_generator_int_double.cu
  src/cuvs_runtime/random/rmat_rectangular_generator_int_float.cu
  src/spatial/knn/detail/ball_cover/registers_pass_one_2d_dist.cu
  src/spatial/knn/detail/ball_cover/registers_pass_one_2d_euclidean.cu
  src/spatial/knn/detail/ball_cover/registers_pass_one_2d_haversine.cu
  src/spatial/knn/detail/ball_cover/registers_pass_one_3d_dist.cu
  src/spatial/knn/detail/ball_cover/registers_pass_one_3d_euclidean.cu
  src/spatial/knn/detail/ball_cover/registers_pass_one_3d_haversine.cu
  src/spatial/knn/detail/ball_cover/registers_pass_two_2d_dist.cu
  src/spatial/knn/detail/ball_cover/registers_pass_two_2d_euclidean.cu
  src/spatial/knn/detail/ball_cover/registers_pass_two_2d_haversine.cu
  src/spatial/knn/detail/ball_cover/registers_pass_two_3d_dist.cu
  src/spatial/knn/detail/ball_cover/registers_pass_two_3d_euclidean.cu
  src/spatial/knn/detail/ball_cover/registers_pass_two_3d_haversine.cu
  src/spatial/knn/detail/fused_l2_knn_int32_t_float.cu
  src/spatial/knn/detail/fused_l2_knn_int64_t_float.cu
  src/spatial/knn/detail/fused_l2_knn_uint32_t_float.cu
)

target_compile_options(
  cuvs INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda
                 --expt-relaxed-constexpr>
)

add_library(cuvs::cuvs ALIAS cuvs)

target_include_directories(
  cuvs PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
              "$<INSTALL_INTERFACE:include>"
)

if(NOT BUILD_CPU_ONLY)
  # Keep cuVS as lightweight as possible. Only CUDA libs and rmm should be used in global target.
  target_link_libraries(cuvs PUBLIC raft::raft)
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
  target_link_libraries(cuvs PUBLIC CUDA::nvToolsExt)
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

target_link_libraries(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:CUDA::nvToolsExt>)
target_compile_definitions(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>)

  ]=]
  )
endif()

set_target_properties(
  cuvs
  PROPERTIES CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 17
             CUDA_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
  cuvs PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
               "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
)
# ensure CUDA symbols aren't relocated to the middle of the debug build binaries
target_link_options(cuvs PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

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

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuvs/version_config.hpp
  COMPONENT cuvs
  DESTINATION include/cuvs
)

# Use `rapids_export` for 22.04 as it will have COMPONENT support
rapids_export(
  INSTALL cuvs
  EXPORT_SET cuvs-exports
  GLOBAL_TARGETS cuvs
  NAMESPACE cuvs::
)

# ##################################################################################################
# * build export -------------------------------------------------------------
rapids_export(
  BUILD cuvs
  EXPORT_SET cuvs-exports
  GLOBAL_TARGETS cuvs
  NAMESPACE cuvs::
)

# ##################################################################################################
# * shared test/bench headers ------------------------------------------------

if(BUILD_TESTS OR BUILD_MICRO_BENCH)
  include(internal/CMakeLists.txt)
endif()

# ##################################################################################################
# * build test executable ----------------------------------------------------

if(BUILD_TESTS)
  include(test/CMakeLists.txt)
endif()

# ##################################################################################################
# * build benchmark executable -----------------------------------------------

if(BUILD_MICRO_BENCH)
  include(bench/micro/CMakeLists.txt)
endif()

# ##################################################################################################
# * build ann benchmark executable -----------------------------------------------

if(BUILD_ANN_BENCH)
  include(bench/ann/CMakeLists.txt)
endif()
