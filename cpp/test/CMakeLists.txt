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

  target_compile_definitions(${TEST_NAME} PRIVATE "CUVS_EXPLICIT_INSTANTIATE_ONLY")

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

if(BUILD_TESTS)
  ConfigureTest(
    NAME NEIGHBORS_TEST PATH neighbors/brute_force.cu neighbors/brute_force_prefiltered.cu
    neighbors/refine.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME CLUSTER_TEST PATH cluster/kmeans.cu cluster/kmeans_balanced.cu cluster/kmeans_find_k.cu
    cluster/linkage.cu GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_IVF_FLAT_TEST
    PATH
    neighbors/ann_ivf_flat/test_float_int64_t.cu
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
    NAME
    NEIGHBORS_ANN_CAGRA_TEST
    PATH
    neighbors/ann_cagra/test_float_uint32_t.cu
    neighbors/ann_cagra/test_int8_t_uint32_t.cu
    neighbors/ann_cagra/test_uint8_t_uint32_t.cu
    GPUS
    1
    PERCENT
    100
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

  if(BUILD_CAGRA_HNSWLIB)
    ConfigureTest(NAME NEIGHBORS_HNSW_TEST PATH neighbors/hnsw.cu GPUS 1 PERCENT 100)
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
    distance/masked_nn.cu
    sparse/neighbors/cross_component_nn.cu
    GPUS
    1
    PERCENT
    100
  )
  ConfigureTest(
    NAME STATS_TEST PATH stats/trustworthiness.cu stats/silhouette_score.cu GPUS 1 PERCENT 100
  )
endif()

if(BUILD_C_TESTS)
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
  endif()
endif()

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)

if(BUILD_C_TESTS)
  enable_language(C)

  add_executable(cuvs_c_test core/c_api.c)
  target_link_libraries(cuvs_c_test PUBLIC cuvs::c_api)

  add_executable(cuvs_c_neighbors_test neighbors/c_api.c)
  target_link_libraries(cuvs_c_neighbors_test PUBLIC cuvs::c_api)
endif()
