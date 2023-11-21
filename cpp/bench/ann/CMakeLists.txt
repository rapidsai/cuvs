# =============================================================================
# Copyright (c) 2023, NVIDIA CORPORATION.
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
# * benchmark options ------------------------------------------------------------------------------

option(CUVS_BENCH_USE_FAISS_GPU_FLAT "Include faiss' brute-force knn algorithm in benchmark" ON)
option(CUVS_BENCH_USE_FAISS_GPU_IVF_FLAT "Include faiss' ivf flat algorithm in benchmark" ON)
option(CUVS_BENCH_USE_FAISS_GPU_IVF_PQ "Include faiss' ivf pq algorithm in benchmark" ON)
option(CUVS_BENCH_USE_FAISS_CPU_FLAT "Include faiss' cpu brute-force knn algorithm in benchmark" ON)
option(CUVS_BENCH_USE_FAISS_CPU_FLAT "Include faiss' cpu brute-force algorithm in benchmark" ON)

option(CUVS_BENCH_USE_FAISS_CPU_IVF_FLAT "Include faiss' cpu ivf flat algorithm in benchmark" ON)
option(CUVS_BENCH_USE_FAISS_CPU_IVF_PQ "Include faiss' cpu ivf pq algorithm in benchmark" ON)
option(CUVS_BENCH_USE_RAFT_IVF_FLAT "Include raft's ivf flat algorithm in benchmark" ON)
option(CUVS_BENCH_USE_RAFT_IVF_PQ "Include raft's ivf pq algorithm in benchmark" ON)
option(CUVS_BENCH_USE_RAFT_CAGRA "Include raft's CAGRA in benchmark" ON)
option(CUVS_BENCH_USE_RAFT_CAGRA_HNSWLIB "Include raft's CAGRA in benchmark" ON)
option(CUVS_BENCH_USE_HNSWLIB "Include hnsw algorithm in benchmark" ON)
option(CUVS_BENCH_USE_GGNN "Include ggnn algorithm in benchmark" ON)
option(CUVS_BENCH_SINGLE_EXE "Make a single executable with benchmark as shared library modules"
       OFF
)

# ##################################################################################################
# * Process options ----------------------------------------------------------

find_package(Threads REQUIRED)

if(BUILD_CPU_ONLY)

  # Include necessary logging dependencies
  include(cmake/thirdparty/get_fmt.cmake)
  include(cmake/thirdparty/get_spdlog.cmake)

  set(CUVS_FAISS_ENABLE_GPU OFF)
  set(CUVS_BENCH_USE_FAISS_GPU_FLAT OFF)
  set(CUVS_BENCH_USE_FAISS_GPU_IVF_FLAT OFF)
  set(CUVS_BENCH_USE_FAISS_GPU_IVF_PQ OFF)
  set(CUVS_BENCH_USE_RAFT_IVF_FLAT OFF)
  set(CUVS_BENCH_USE_RAFT_IVF_PQ OFF)
  set(CUVS_BENCH_USE_RAFT_CAGRA OFF)
  set(CUVS_BENCH_USE_RAFT_CAGRA_HNSWLIB OFF)
  set(CUVS_BENCH_USE_GGNN OFF)
else()
  # Disable faiss benchmarks on CUDA 12 since faiss is not yet CUDA 12-enabled.
  # https://github.com/rapidsai/raft/issues/1627
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0.0)
    set(CUVS_FAISS_ENABLE_GPU OFF)
    set(CUVS_BENCH_USE_FAISS_GPU_FLAT OFF)
    set(CUVS_BENCH_USE_FAISS_GPU_IVF_FLAT OFF)
    set(CUVS_BENCH_USE_FAISS_GPU_IVF_PQ OFF)
    set(CUVS_BENCH_USE_FAISS_CPU_FLAT OFF)
    set(CUVS_BENCH_USE_FAISS_CPU_IVF_PQ OFF)
    set(CUVS_BENCH_USE_FAISS_CPU_IVF_FLAT OFF)
  else()
    set(CUVS_FAISS_ENABLE_GPU ON)
  endif()
endif()

set(CUVS_BENCH_USE_FAISS OFF)
if(CUVS_BENCH_USE_FAISS_GPU_FLAT
   OR CUVS_BENCH_USE_FAISS_GPU_IVF_PQ
   OR CUVS_BENCH_USE_FAISS_GPU_IVF_FLAT
   OR CUVS_BENCH_USE_FAISS_CPU_FLAT
   OR CUVS_BENCH_USE_FAISS_CPU_IVF_PQ
   OR CUVS_BENCH_USE_FAISS_CPU_IVF_FLAT
)
  set(CUVS_BENCH_USE_FAISS ON)
  set(RAFT_USE_FAISS_STATIC ON)
endif()

set(CUVS_BENCH_USE_RAFT OFF)
if(CUVS_BENCH_USE_RAFT_IVF_PQ
   OR CUVS_BENCH_USE_RAFT_BRUTE_FORCE
   OR CUVS_BENCH_USE_RAFT_IVF_FLAT
   OR CUVS_BENCH_USE_RAFT_CAGRA
   OR CUVS_BENCH_USE_RAFT_CAGRA_HNSWLIB
)
  set(CUVS_BENCH_USE_RAFT ON)
endif()

# ##################################################################################################
# * Fetch requirements -------------------------------------------------------------

if(CUVS_BENCH_USE_HNSWLIB OR CUVS_BENCH_USE_RAFT_CAGRA_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib.cmake)
endif()

include(cmake/thirdparty/get_nlohmann_json.cmake)

if(CUVS_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_ggnn.cmake)
endif()

if(CUVS_BENCH_USE_FAISS)
  # We need to ensure that faiss has all the conda information. So we currently use the very ugly
  # hammer of `link_libraries` to ensure that all targets in this directory and the faiss directory
  # will have the conda includes/link dirs
  link_libraries($<TARGET_NAME_IF_EXISTS:conda_env>)
  include(cmake/thirdparty/get_faiss.cmake)
endif()

# ##################################################################################################
# * Configure tests function-------------------------------------------------------------

function(ConfigureAnnBench)

  set(oneValueArgs NAME)
  set(multiValueArgs PATH LINKS CXXFLAGS INCLUDES)

  if(NOT BUILD_CPU_ONLY)
    set(GPU_BUILD ON)
  endif()

  cmake_parse_arguments(
    ConfigureAnnBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  set(BENCH_NAME ${ConfigureAnnBench_NAME}_CUVS_BENCH)

  if(CUVS_BENCH_SINGLE_EXE)
    add_library(${BENCH_NAME} SHARED ${ConfigureAnnBench_PATH})
    string(TOLOWER ${BENCH_NAME} BENCH_LIB_NAME)
    set_target_properties(${BENCH_NAME} PROPERTIES OUTPUT_NAME ${BENCH_LIB_NAME})
    add_dependencies(${BENCH_NAME} CUVS_BENCH)
  else()
    add_executable(${BENCH_NAME} ${ConfigureAnnBench_PATH})
    target_compile_definitions(${BENCH_NAME} PRIVATE CUVS_BENCH_BUILD_MAIN)
    target_link_libraries(${BENCH_NAME} PRIVATE benchmark::benchmark)
  endif()

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE raft::raft
            nlohmann_json::nlohmann_json
            ${ConfigureAnnBench_LINKS}
            Threads::Threads
            $<$<BOOL:${GPU_BUILD}>:${RAFT_CTK_MATH_DEPENDENCIES}>
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
            -static-libgcc
            -static-libstdc++
            $<$<BOOL:${BUILD_CPU_ONLY}>:fmt::fmt-header-only>
            $<$<BOOL:${BUILD_CPU_ONLY}>:spdlog::spdlog_header_only>
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

  set(${ConfigureAnnBench_CXXFLAGS} ${RAFT_CXX_FLAGS} ${ConfigureAnnBench_CXXFLAGS})

  target_compile_options(
    ${BENCH_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ConfigureAnnBench_CXXFLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )

  if(CUVS_BENCH_USE_${ConfigureAnnBench_NAME})
    target_compile_definitions(
      ${BENCH_NAME}
      PUBLIC CUVS_BENCH_USE_${ConfigureAnnBench_NAME}=CUVS_BENCH_USE_${ConfigureAnnBench_NAME}
    )
  endif()

  target_include_directories(
    ${BENCH_NAME}
    PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>"
    PRIVATE ${ConfigureAnnBench_INCLUDES}
  )

  install(
    TARGETS ${BENCH_NAME}
    COMPONENT ann_bench
    DESTINATION bin/ann
  )
endfunction()

# ##################################################################################################
# * Configure tests-------------------------------------------------------------

if(CUVS_BENCH_USE_HNSWLIB)
  ConfigureAnnBench(
    NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp INCLUDES
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib CXXFLAGS "${HNSW_CXX_FLAGS}"
  )
endif()

if(CUVS_BENCH_USE_RAFT_IVF_PQ)
  ConfigureAnnBench(
    NAME RAFT_IVF_PQ PATH bench/ann/src/raft/raft_benchmark.cu
    $<$<BOOL:${CUVS_BENCH_USE_RAFT_IVF_PQ}>:bench/ann/src/raft/raft_ivf_pq.cu> LINKS cuvs
  )
endif()

if(CUVS_BENCH_USE_RAFT_IVF_FLAT)
  ConfigureAnnBench(
    NAME RAFT_IVF_FLAT PATH bench/ann/src/raft/raft_benchmark.cu
    $<$<BOOL:${CUVS_BENCH_USE_RAFT_IVF_FLAT}>:bench/ann/src/raft/raft_ivf_flat.cu> LINKS cuvs
  )
endif()

if(CUVS_BENCH_USE_RAFT_BRUTE_FORCE)
  ConfigureAnnBench(NAME RAFT_BRUTE_FORCE PATH bench/ann/src/raft/raft_benchmark.cu LINKS cuvs)
endif()

if(CUVS_BENCH_USE_RAFT_CAGRA)
  ConfigureAnnBench(
    NAME RAFT_CAGRA PATH bench/ann/src/raft/raft_benchmark.cu
    $<$<BOOL:${CUVS_BENCH_USE_RAFT_CAGRA}>:bench/ann/src/raft/raft_cagra.cu> LINKS cuvs
  )
endif()

if(CUVS_BENCH_USE_RAFT_CAGRA_HNSWLIB)
  ConfigureAnnBench(
    NAME RAFT_CAGRA_HNSWLIB PATH bench/ann/src/raft/raft_cagra_hnswlib.cu INCLUDES
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/hnswlib-src/hnswlib LINKS cuvs CXXFLAGS "${HNSW_CXX_FLAGS}"
  )
endif()

set(RAFT_FAISS_TARGETS faiss::faiss)
if(TARGET faiss::faiss_avx2)
  set(RAFT_FAISS_TARGETS faiss::faiss_avx2)
endif()

message("RAFT_FAISS_TARGETS: ${RAFT_FAISS_TARGETS}")
message("CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
if(CUVS_BENCH_USE_FAISS_CPU_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_FLAT PATH bench/ann/src/faiss/faiss_cpu_benchmark.cpp LINKS
    ${RAFT_FAISS_TARGETS}
  )
endif()

if(CUVS_BENCH_USE_FAISS_CPU_IVF_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_FLAT PATH bench/ann/src/faiss/faiss_cpu_benchmark.cpp LINKS
    ${RAFT_FAISS_TARGETS}
  )
endif()

if(CUVS_BENCH_USE_FAISS_CPU_IVF_PQ)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_PQ PATH bench/ann/src/faiss/faiss_cpu_benchmark.cpp LINKS
    ${RAFT_FAISS_TARGETS}
  )
endif()

if(CUVS_BENCH_USE_FAISS_GPU_IVF_FLAT)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_FLAT PATH bench/ann/src/faiss/faiss_gpu_benchmark.cu LINKS
    ${RAFT_FAISS_TARGETS}
  )
endif()

if(CUVS_BENCH_USE_FAISS_GPU_IVF_PQ)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_PQ PATH bench/ann/src/faiss/faiss_gpu_benchmark.cu LINKS
    ${RAFT_FAISS_TARGETS}
  )
endif()

if(CUVS_BENCH_USE_FAISS_GPU_FLAT)
  ConfigureAnnBench(
    NAME FAISS_GPU_FLAT PATH bench/ann/src/faiss/faiss_gpu_benchmark.cu LINKS ${RAFT_FAISS_TARGETS}
  )
endif()

if(CUVS_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_glog.cmake)
  ConfigureAnnBench(
    NAME GGNN PATH bench/ann/src/ggnn/ggnn_benchmark.cu INCLUDES
    ${CMAKE_CURRENT_BINARY_DIR}/_deps/ggnn-src/include LINKS glog::glog
  )
endif()

# ##################################################################################################
# * Dynamically-loading CUVS_BENCH executable
# -------------------------------------------------------
if(CUVS_BENCH_SINGLE_EXE)
  add_executable(CUVS_BENCH bench/ann/src/common/benchmark.cpp)

  # Build and link static version of the GBench to keep CUVS_BENCH self-contained.
  get_target_property(TMP_PROP benchmark::benchmark SOURCES)
  add_library(benchmark_static STATIC ${TMP_PROP})
  get_target_property(TMP_PROP benchmark::benchmark INCLUDE_DIRECTORIES)
  target_include_directories(benchmark_static PUBLIC ${TMP_PROP})
  get_target_property(TMP_PROP benchmark::benchmark LINK_LIBRARIES)
  target_link_libraries(benchmark_static PUBLIC ${TMP_PROP})

  target_include_directories(CUVS_BENCH PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(
    CUVS_BENCH PRIVATE nlohmann_json::nlohmann_json benchmark_static dl -static-libgcc
                       -static-libstdc++ CUDA::nvtx3
  )
  set_target_properties(
    CUVS_BENCH
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

  # Disable NVTX when the nvtx3 headers are missing
  set(_CMAKE_REQUIRED_INCLUDES_ORIG ${CMAKE_REQUIRED_INCLUDES})
  get_target_property(CMAKE_REQUIRED_INCLUDES CUVS_BENCH INCLUDE_DIRECTORIES)
  CHECK_INCLUDE_FILE_CXX(nvtx3/nvToolsExt.h NVTX3_HEADERS_FOUND)
  set(CMAKE_REQUIRED_INCLUDES ${_CMAKE_REQUIRED_INCLUDES_ORIG})
  target_compile_definitions(
    CUVS_BENCH
    PRIVATE
      $<$<BOOL:${CUDAToolkit_FOUND}>:CUVS_BENCH_LINK_CUDART="libcudart.so.${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}.${CUDAToolkit_VERSION_PATCH}
  ">
      $<$<BOOL:${NVTX3_HEADERS_FOUND}>:CUVS_BENCH_NVTX3_HEADERS_FOUND>
  )

  target_link_options(CUVS_BENCH PRIVATE -export-dynamic)

  install(
    TARGETS CUVS_BENCH
    COMPONENT ann_bench
    DESTINATION bin/ann
    EXCLUDE_FROM_ALL
  )
endif()
