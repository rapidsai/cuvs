# =============================================================================
# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

  set(options OPTIONAL NOCUDA C_LIB)
  set(oneValueArgs NAME GPUS PERCENT ADDITIONAL_DEP)
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
    PRIVATE cuvs
            cuvs::cuvs
            raft::raft
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
            ${CUVS_CTK_MATH_DEPENDENCIES}
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
            $<$<BOOL:${_CUVS_TEST_C_LIB}>:cuvs::c_api>
            ${_CUVS_TEST_ADDITIONAL_DEP}
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_BINARY_DIR}/gtests>"
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

  if(_CUVS_TEST_NOCUDA)
    target_compile_definitions(${TEST_NAME} PRIVATE "CUVS_DISABLE_CUDA")
  endif()

  target_include_directories(${TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/tests>")

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

if(BUILD_TESTS)
  ConfigureTest(
    NAME NEIGHBORS_TEST PATH neighbors/brute_force.cu neighbors/brute_force_prefiltered.cu
    neighbors/sparse_brute_force.cu neighbors/refine.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(NAME NEIGHBORS_TIERED_INDEX_TEST PATH neighbors/tiered_index.cu GPUS 1 PERCENT 100)

  ConfigureTest(
    NAME CLUSTER_TEST PATH cluster/kmeans.cu cluster/kmeans_balanced.cu cluster/kmeans_find_k.cu
    cluster/linkage.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_IVF_FLAT_TEST
    PATH
    neighbors/ann_ivf_flat/test_float_int64_t.cu
    neighbors/ann_ivf_flat/test_half_int64_t.cu
    neighbors/ann_ivf_flat/test_int8_t_int64_t.cu
    neighbors/ann_ivf_flat/test_uint8_t_int64_t.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_IVF_PQ_TEST
    PATH
    neighbors/ann_ivf_pq/test_float_int64_t.cu
    neighbors/ann_ivf_pq/test_int8_t_int64_t.cu
    neighbors/ann_ivf_pq/test_uint8_t_int64_t.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_BRUTE_FORCE_TEST PATH neighbors/ann_brute_force/test_float.cu
    neighbors/ann_brute_force/test_half.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_CAGRA_TEST_BUGS PATH neighbors/ann_cagra/bug_extreme_inputs_oob.cu
    neighbors/ann_cagra/bug_multi_cta_crash.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_CAGRA_FLOAT_UINT32_TEST PATH neighbors/ann_cagra/test_float_uint32_t.cu GPUS
    1 PERCENT 100
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_CAGRA_HALF_UINT32_TEST PATH neighbors/ann_cagra/test_half_uint32_t.cu GPUS 1
    PERCENT 100
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_CAGRA_INT8_UINT32_TEST PATH neighbors/ann_cagra/test_int8_t_uint32_t.cu GPUS
    1 PERCENT 100
  )

  ConfigureTest(
    NAME NEIGHBORS_ANN_CAGRA_UINT8_UINT32_TEST PATH neighbors/ann_cagra/test_uint8_t_uint32_t.cu
    GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_NN_DESCENT_TEST
    PATH
    neighbors/ann_nn_descent/test_float_uint32_t.cu
    neighbors/ann_nn_descent/test_int8_t_uint32_t.cu
    neighbors/ann_nn_descent/test_uint8_t_uint32_t.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_VAMANA_TEST
    PATH
    neighbors/ann_vamana/test_float_uint32_t.cu
    neighbors/ann_vamana/test_int8_t_uint32_t.cu
    neighbors/ann_vamana/test_uint8_t_uint32_t.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(NAME NEIGHBORS_BALL_COVER_TEST PATH neighbors/ball_cover.cu GPUS 1 PERCENT 100)

  ConfigureTest(
    NAME NEIGHBORS_ALL_NEIGHBORS_TEST PATH neighbors/all_neighbors/test_float.cu GPUS 1 PERCENT 100
    ADDITIONAL_DEP nccl
  )

  ConfigureTest(
    NAME
    NEIGHBORS_DYNAMIC_BATCHING_TEST
    PATH
    neighbors/dynamic_batching/test_brute_force.cu
    neighbors/dynamic_batching/test_cagra.cu
    neighbors/dynamic_batching/test_ivf_flat.cu
    neighbors/dynamic_batching/test_ivf_pq.cu
    GPUS
    1
    PERCENT
    100
  )

  if(BUILD_CAGRA_HNSWLIB)
    ConfigureTest(NAME NEIGHBORS_HNSW_TEST PATH neighbors/hnsw.cu GPUS 1 PERCENT 100)
    target_link_libraries(NEIGHBORS_HNSW_TEST PRIVATE hnswlib::hnswlib)
    target_compile_definitions(NEIGHBORS_HNSW_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB)
  endif()

  if(BUILD_MG_ALGOS)
    ConfigureTest(
      NAME NEIGHBORS_MG_TEST PATH neighbors/mg/test_float.cu GPUS 1 PERCENT 100 ADDITIONAL_DEP
      NCCL::NCCL
    )
  endif()

  ConfigureTest(
    NAME
    DISTANCE_TEST
    PATH
    distance/dist_canberra.cu
    distance/dist_correlation.cu
    distance/dist_cos.cu
    distance/dist_hamming.cu
    distance/dist_hellinger.cu
    distance/dist_inner_product.cu
    distance/dist_jensen_shannon.cu
    distance/dist_kl_divergence.cu
    distance/dist_l1.cu
    distance/dist_l2_exp.cu
    distance/dist_l2_sqrt_exp.cu
    distance/dist_l_inf.cu
    distance/dist_lp_unexp.cu
    distance/dist_russell_rao.cu
    distance/gram.cu
    distance/masked_nn.cu
    distance/sparse_distance.cu
    sparse/neighbors/cross_component_nn.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME SPARSE_TEST PATH sparse/cluster/cluster_solvers.cu sparse/cluster/eigen_solvers.cu
    sparse/cluster/spectral.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME PREPROCESSING_TEST PATH preprocessing/scalar_quantization.cu
    preprocessing/binary_quantization.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME STATS_TEST PATH stats/trustworthiness.cu stats/silhouette_score.cu GPUS 1 PERCENT 100
  )
endif()

if(TARGET cuvs::c_api)
  enable_language(C)

  ConfigureTest(NAME INTEROP_TEST PATH core/interop.cu C_LIB)
  ConfigureTest(
    NAME DISTANCE_C_TEST PATH distance/run_pairwise_distance_c.c distance/pairwise_distance_c.cu
    C_LIB
  )

  ConfigureTest(
    NAME BRUTEFORCE_C_TEST PATH neighbors/run_brute_force_c.c neighbors/brute_force_c.cu C_LIB
  )

  ConfigureTest(
    NAME IVF_FLAT_C_TEST PATH neighbors/run_ivf_flat_c.c neighbors/ann_ivf_flat_c.cu C_LIB
  )

  ConfigureTest(NAME IVF_PQ_C_TEST PATH neighbors/run_ivf_pq_c.c neighbors/ann_ivf_pq_c.cu C_LIB)

  ConfigureTest(NAME CAGRA_C_TEST PATH neighbors/ann_cagra_c.cu C_LIB)

  if(BUILD_CAGRA_HNSWLIB)
    ConfigureTest(NAME HNSW_C_TEST PATH neighbors/ann_hnsw_c.cu C_LIB)
    target_link_libraries(NEIGHBORS_HNSW_TEST PRIVATE hnswlib::hnswlib)
    target_compile_definitions(NEIGHBORS_HNSW_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB)
  endif()

  add_executable(cuvs_c_test core/c_api.c)
  target_link_libraries(cuvs_c_test PUBLIC cuvs::c_api)

  add_executable(cuvs_c_neighbors_test neighbors/c_api.c)
  target_link_libraries(cuvs_c_neighbors_test PUBLIC cuvs::c_api)

  set_target_properties(
    cuvs_c_test cuvs_c_neighbors_test
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )

  rapids_test_add(
    NAME cuvs_c_test
    COMMAND cuvs_c_test
    GPUS 1
    PERCENT 100
    INSTALL_COMPONENT_SET testing
  )
  rapids_test_add(
    NAME cuvs_c_neighbors_test
    COMMAND cuvs_c_neighbors_test
    GPUS 1
    PERCENT 100
    INSTALL_COMPONENT_SET testing
  )
endif()

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)
