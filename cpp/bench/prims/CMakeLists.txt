# =============================================================================
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

list(APPEND CMAKE_MODULE_PATH "${CUVS_SOURCE_DIR}")

# ##################################################################################################
# * benchmark options ------------------------------------------------------------------------------

option(CUVS_PRIMS_BENCH_USE_FUSED_L2_NN "Include Fused L2 NN benchmark" ON)
option(CUVS_PRIMS_BENCH_USE_CUBLAS_SAMPLE "Include cuBLAS sample benchmark" ON)

# ##################################################################################################
# * Process options ----------------------------------------------------------

find_package(Threads REQUIRED)

# ##################################################################################################
# * Fetch requirements -------------------------------------------------------------

include(cmake/thirdparty/get_nlohmann_json)

# ##################################################################################################
# * Target function -------------------------------------------------------------

function(ConfigurePrimsBench)

  set(oneValueArgs NAME)
  set(multiValueArgs PATH LINKS CXXFLAGS)

  #if(NOT BUILD_CPU_ONLY)
    set(GPU_BUILD ON)
  #endif()

  cmake_parse_arguments(
    ConfigurePrimsBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  set(BENCH_NAME ${ConfigurePrimsBench_NAME}_PRIMS_BENCH)

  add_executable(${BENCH_NAME} ${ConfigurePrimsBench_PATH})
  target_compile_definitions(${BENCH_NAME} PRIVATE PRIMS_BENCH_BUILD_MAIN)
  target_link_libraries(
    ${BENCH_NAME} PRIVATE benchmark::benchmark $<$<TARGET_EXISTS:CUDA::nvtx3>:CUDA::nvtx3>
  )

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE ${ConfigurePrimsBench_LINKS}
            nlohmann_json::nlohmann_json
            Threads::Threads
            $<$<BOOL:${GPU_BUILD}>:CUDA::cudart_static>
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )

  set_target_properties(
    ${BENCH_NAME}
    PROPERTIES # set target compile options
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
  )

  set(${ConfigurePrimsBench_CXXFLAGS} ${CUVS_CXX_FLAGS} ${ConfigurePrimsBench_CXXFLAGS})

  target_compile_options(
    ${BENCH_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ConfigurePrimsBench_CXXFLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
  )

  if(CUVS_PRIMS_BENCH_USE_${ConfigurePrimsBench_NAME})
    target_compile_definitions(
      ${BENCH_NAME}
      PUBLIC
      CUVS_PRIMS_BENCH_USE_${ConfigurePrimsBench_NAME}=CUVS_PRIMS_BENCH_USE_${ConfigurePrimsBench_NAME}
    )
  endif()

  target_include_directories(
    ${BENCH_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
    PRIVATE ${ConfigurePrimsBench_INCLUDES}
  )

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT prims_bench
    DESTINATION bin/prims
  )

  add_dependencies(CUVS_PRIMS_BENCH_ALL ${BENCH_NAME})
endfunction()

# ##################################################################################################
# * Configure benchmark targets -------------------------------------------------------------

if(NOT TARGET CUVS_PRIMS_BENCH_ALL)
  add_custom_target(CUVS_PRIMS_BENCH_ALL)
endif()

if(CUVS_PRIMS_BENCH_USE_FUSED_L2_NN)
  ConfigurePrimsBench(
    NAME FUSED_L2_NN PATH src/distance/fused_l2_nn.cu
  )
endif()

if(CUVS_PRIMS_BENCH_USE_CUBLAS_SAMPLE)
  ConfigurePrimsBench(
    NAME CUBLAS_GEMM PATH src/distance/cublas_sample.cu
    LINKS CUDA::cublas
  )
endif()

message("CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")

