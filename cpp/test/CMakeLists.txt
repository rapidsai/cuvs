# =============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
# enable testing ################################################################################
# ##################################################################################################
enable_testing()
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY NOCUDA)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH TARGETS CONFIGURATIONS)

  cmake_parse_arguments(_CUVS_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED _CUVS_TEST_GPUS AND NOT DEFINED _CUVS_TEST_PERCENT)
    set(_CUVS_TEST_GPUS 1)
    set(_CUVS_TEST_PERCENT 30)
  endif()
  if(NOT DEFINED _CUVS_TEST_GPUS)
    set(_CUVS_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _CUVS_TEST_PERCENT)
    set(_CUVS_TEST_PERCENT 100)
  endif()

  if(_CUVS_TEST_NOCUDA)
    set(TEST_NAME "${_CUVS_TEST_NAME}_NOCUDA")
  else()
    set(TEST_NAME ${_CUVS_TEST_NAME})
  endif()

  add_executable(${TEST_NAME} ${_CUVS_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE cuvs raft::raft cuvs_internal GTest::gtest GTest::gtest_main Threads::Threads
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX> $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    ${TEST_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
  )
  if(_CUVS_TEST_EXPLICIT_INSTANTIATE_ONLY)
    target_compile_definitions(${TEST_NAME} PRIVATE "CUVS_EXPLICIT_INSTANTIATE_ONLY")
  endif()
  if(_CUVS_TEST_NOCUDA)
    target_compile_definitions(${TEST_NAME} PRIVATE "CUVS_DISABLE_CUDA")
  endif()

  target_include_directories(${TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/test>")

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_CUVS_TEST_GPUS}
    PERCENT ${_CUVS_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################

# ##################################################################################################
# * distance tests -------------------------------------------------------------------------

if(BUILD_TESTS)
  ConfigureTest(
    NAME
    CLUSTER_TEST
    PATH
    test/cluster/kmeans.cu
    test/cluster/kmeans_balanced.cu
    test/cluster/cluster_solvers.cu
    test/cluster/linkage.cu
    test/cluster/kmeans_find_k.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    DISTANCE_TEST
    PATH
    test/distance/dist_adj.cu
    test/distance/dist_adj_distance_instance.cu
    test/distance/dist_canberra.cu
    test/distance/dist_correlation.cu
    test/distance/dist_cos.cu
    test/distance/dist_hamming.cu
    test/distance/dist_hellinger.cu
    test/distance/dist_inner_product.cu
    test/distance/dist_jensen_shannon.cu
    test/distance/dist_kl_divergence.cu
    test/distance/dist_l1.cu
    test/distance/dist_l2_exp.cu
    test/distance/dist_l2_unexp.cu
    test/distance/dist_l2_sqrt_exp.cu
    test/distance/dist_l_inf.cu
    test/distance/dist_lp_unexp.cu
    test/distance/dist_russell_rao.cu
    test/distance/masked_nn.cu
    test/distance/masked_nn_compress_to_bits.cu
    test/distance/fused_l2_nn.cu
    test/distance/gram.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    NEIGHBORS_TEST
    PATH
    test/neighbors/knn.cu
    test/neighbors/fused_l2_knn.cu
    test/neighbors/tiled_knn.cu
    test/neighbors/haversine.cu
    test/neighbors/ball_cover.cu
    test/neighbors/epsilon_neighborhood.cu
    test/neighbors/refine.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_CAGRA_TEST
    PATH
    test/neighbors/ann_cagra/test_float_uint32_t.cu
    test/neighbors/ann_cagra/test_int8_t_uint32_t.cu
    test/neighbors/ann_cagra/test_uint8_t_uint32_t.cu
    test/neighbors/ann_cagra/test_float_int64_t.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim128_t8.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim256_t16.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim512_t32.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim1024_t32.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim128_t8.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim256_t16.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim512_t32.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim1024_t32.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_IVF_TEST
    PATH
    test/neighbors/ann_ivf_flat/test_filter_float_int64_t.cu
    test/neighbors/ann_ivf_flat/test_float_int64_t.cu
    test/neighbors/ann_ivf_flat/test_int8_t_int64_t.cu
    test/neighbors/ann_ivf_flat/test_uint8_t_int64_t.cu
    test/neighbors/ann_ivf_pq/test_float_uint32_t.cu
    test/neighbors/ann_ivf_pq/test_float_int64_t.cu
    test/neighbors/ann_ivf_pq/test_int8_t_int64_t.cu
    test/neighbors/ann_ivf_pq/test_uint8_t_int64_t.cu
    test/neighbors/ann_ivf_pq/test_filter_float_int64_t.cu
    test/neighbors/ann_ivf_pq/test_filter_int8_t_int64_t.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_NN_DESCENT_TEST
    PATH
    test/neighbors/ann_nn_descent/test_float_uint32_t.cu
    test/neighbors/ann_nn_descent/test_int8_t_uint32_t.cu
    test/neighbors/ann_nn_descent/test_uint8_t_uint32_t.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME NEIGHBORS_SELECTION_TEST PATH test/neighbors/selection.cu LIB EXPLICIT_INSTANTIATE_ONLY
    GPUS 1 PERCENT 50
  )

  ConfigureTest(
    NAME
    STATS_TEST
    PATH
    test/stats/contingencyMatrix.cu
    test/stats/cov.cu
    test/stats/dispersion.cu
    test/stats/entropy.cu
    test/stats/histogram.cu
    test/stats/homogeneity_score.cu
    test/stats/information_criterion.cu
    test/stats/kl_divergence.cu
    test/stats/mutual_info_score.cu
    test/stats/neighborhood_recall.cu
    test/stats/r2_score.cu
    test/stats/rand_index.cu
    test/stats/silhouette_score.cu
    test/stats/trustworthiness.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )
endif()

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)
