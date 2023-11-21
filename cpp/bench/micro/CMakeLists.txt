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

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

function(ConfigureBench)

  set(options OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY)
  set(oneValueArgs NAME)
  set(multiValueArgs PATH TARGETS CONFIGURATIONS)

  cmake_parse_arguments(ConfigureBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(BENCH_NAME ${ConfigureBench_NAME})

  add_executable(${BENCH_NAME} ${ConfigureBench_PATH})

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE raft::raft cuvs_internal benchmark::benchmark Threads::Threads
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX> $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  set_target_properties(
    ${BENCH_NAME}
    PROPERTIES # set target compile options
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  target_compile_options(
    ${BENCH_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
  )

  if(ConfigureTest_EXPLICIT_INSTANTIATE_ONLY)
    target_compile_definitions(${BENCH_NAME} PRIVATE "CUVS_EXPLICIT_INSTANTIATE_ONLY")
  endif()

  target_include_directories(
    ${BENCH_NAME} PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/bench/micro>"
  )

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT testing
    DESTINATION bin/gbench/micro/libcuvs
    EXCLUDE_FROM_ALL
  )

endfunction()

if(BUILD_MICRO_BENCH)
  ConfigureBench(
    NAME CLUSTER_BENCH PATH bench/micro/cluster/kmeans_balanced.cu bench/micro/cluster/kmeans.cu
    bench/micro/main.cpp OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureBench(
    NAME TUNE_DISTANCE PATH bench/micro/distance/tune_pairwise/kernel.cu
    bench/micro/distance/tune_pairwise/bench.cu bench/micro/main.cpp
  )

  ConfigureBench(
    NAME
    DISTANCE_BENCH
    PATH
    bench/micro/distance/distance_cosine.cu
    bench/micro/distance/distance_exp_l2.cu
    bench/micro/distance/distance_l1.cu
    bench/micro/distance/distance_unexp_l2.cu
    bench/micro/distance/fused_l2_nn.cu
    bench/micro/distance/masked_nn.cu
    bench/micro/distance/kernels.cu
    bench/micro/main.cpp
    OPTIONAL
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureBench(
    NAME
    NEIGHBORS_BENCH
    PATH
    bench/micro/neighbors/knn/brute_force_float_int64_t.cu
    bench/micro/neighbors/knn/brute_force_float_uint32_t.cu
    bench/micro/neighbors/knn/cagra_float_uint32_t.cu
    bench/micro/neighbors/knn/ivf_flat_filter_float_int64_t.cu
    bench/micro/neighbors/knn/ivf_flat_float_int64_t.cu
    bench/micro/neighbors/knn/ivf_flat_int8_t_int64_t.cu
    bench/micro/neighbors/knn/ivf_flat_uint8_t_int64_t.cu
    bench/micro/neighbors/knn/ivf_pq_float_int64_t.cu
    bench/micro/neighbors/knn/ivf_pq_filter_float_int64_t.cu
    bench/micro/neighbors/knn/ivf_pq_int8_t_int64_t.cu
    bench/micro/neighbors/knn/ivf_pq_uint8_t_int64_t.cu
    bench/micro/neighbors/refine_float_int64_t.cu
    bench/micro/neighbors/refine_uint8_t_int64_t.cu
    bench/micro/main.cpp
    OPTIONAL
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

endif()
