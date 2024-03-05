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
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
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
  ConfigureTest(NAME NEIGHBORS_TEST PATH test/neighbors/brute_force.cu GPUS 1 PERCENT 100)

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_IVF_FLAT_TEST
    PATH
    test/neighbors/ann_ivf_flat/test_float_int64_t.cu
    test/neighbors/ann_ivf_flat/test_int8_t_int64_t.cu
    test/neighbors/ann_ivf_flat/test_uint8_t_int64_t.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_IVF_PQ_TEST
    PATH
    test/neighbors/ann_ivf_pq/test_float_int64_t.cu
    test/neighbors/ann_ivf_pq/test_int8_t_int64_t.cu
    test/neighbors/ann_ivf_pq/test_uint8_t_int64_t.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_CAGRA_TEST
    PATH
    test/neighbors/ann_cagra/test_float_uint32_t.cu
    test/neighbors/ann_cagra/test_int8_t_uint32_t.cu
    test/neighbors/ann_cagra/test_uint8_t_uint32_t.cu
    GPUS
    1
    PERCENT
    100
  )
endif()

if(BUILD_C_TESTS)
  ConfigureTest(NAME INTEROP_TEST PATH test/core/interop.cu C_LIB)

  ConfigureTest(
    NAME
    BRUTEFORCE_C_TEST
    PATH
    test/neighbors/run_brute_force_c.c
    test/neighbors/brute_force_c.cu
    C_LIB
  )

  ConfigureTest(
    NAME
    IVF_FLAT_C_TEST
    PATH
    test/neighbors/run_ivf_flat_c.c
    test/neighbors/ann_ivf_flat_c.cu
    C_LIB
  )

  ConfigureTest(
    NAME
    IVF_PQ_C_TEST
    PATH
    test/neighbors/run_ivf_pq_c.c
    test/neighbors/ann_ivf_pq_c.cu C_LIB
  )

  ConfigureTest(NAME CAGRA_C_TEST PATH test/neighbors/ann_cagra_c.cu C_LIB)
endif()

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libcuvs)

if(BUILD_C_TESTS)
  enable_language(C)

  add_executable(cuvs_c_test test/core/c_api.c)
  target_link_libraries(cuvs_c_test PUBLIC cuvs::c_api)

  add_executable(cuvs_c_neighbors_test test/neighbors/c_api.c)
  target_link_libraries(cuvs_c_neighbors_test PUBLIC cuvs::c_api)
endif()
