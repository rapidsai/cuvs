# =============================================================================
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

include(rapids-test)

if(PROJECT_IS_TOP_LEVEL)
  rapids_test_init()
endif()

include(${rapids-cmake-dir}/cpm/gtest.cmake)
rapids_cpm_gtest(BUILD_STATIC)

function(ConfigureTest)

  set(options)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH)

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

  set(TEST_NAME ${_CUVS_TEST_NAME})

  add_executable(${TEST_NAME} ${_CUVS_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME} PRIVATE cuvs::c_api GTest::gtest GTest::gtest_main
                         $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_C_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
  )

  target_include_directories(
    ${TEST_NAME} PRIVATE "$<BUILD_INTERFACE:${CUVS_C_SOURCE_DIR}/tests>"
                         "$<BUILD_INTERFACE:${CUVS_C_SOURCE_DIR}/../cpp/tests>"
  )

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_CUVS_TEST_GPUS}
    PERCENT ${_CUVS_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

ConfigureTest(NAME INTEROP_TEST PATH core/interop.cu)
ConfigureTest(
  NAME DISTANCE_C_TEST PATH distance/run_pairwise_distance_c.c distance/pairwise_distance_c.cu
)
ConfigureTest(NAME BRUTEFORCE_C_TEST PATH neighbors/run_brute_force_c.c neighbors/brute_force_c.cu)
ConfigureTest(NAME IVF_FLAT_C_TEST PATH neighbors/run_ivf_flat_c.c neighbors/ann_ivf_flat_c.cu)
ConfigureTest(NAME IVF_PQ_C_TEST PATH neighbors/run_ivf_pq_c.c neighbors/ann_ivf_pq_c.cu)
ConfigureTest(NAME CAGRA_C_TEST PATH neighbors/ann_cagra_c.cu)
ConfigureTest(NAME MG_C_TEST PATH neighbors/run_mg_c.c neighbors/ann_mg_c.cu)
ConfigureTest(
  NAME ALL_NEIGHBORS_C_TEST PATH neighbors/run_all_neighbors_c.c neighbors/all_neighbors_c.cu
)

if(BUILD_CAGRA_HNSWLIB)
  ConfigureTest(NAME HNSW_C_TEST PATH neighbors/ann_hnsw_c.cu)
  target_link_libraries(HNSW_C_TEST PRIVATE hnswlib::hnswlib)
  target_compile_definitions(HNSW_C_TEST PUBLIC CUVS_BUILD_CAGRA_HNSWLIB)
endif()

add_executable(cuvs_c_test core/c_api.c)
target_link_libraries(cuvs_c_test PUBLIC cuvs::c_api)

add_executable(cuvs_c_neighbors_test neighbors/c_api.c)
target_link_libraries(cuvs_c_neighbors_test PUBLIC cuvs::c_api)

set_target_properties(
  cuvs_c_test cuvs_c_neighbors_test
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_C_BINARY_DIR}/gtests>"
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

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
if(PROJECT_IS_TOP_LEVEL)
  rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)
endif()
