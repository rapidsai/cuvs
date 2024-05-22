# =============================================================================
# Copyright (c) 2024, NVIDIA CORPORATION.
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

option(CUVS_ANN_BENCH_USE_FAISS_GPU_FLAT "Include faiss' brute-force knn algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT "Include faiss' ivf flat algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_PQ "Include faiss' ivf pq algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_FLAT "Include faiss' cpu brute-force algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_FLAT "Include faiss' cpu ivf flat algorithm in benchmark"
       OFF
)
option(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_PQ "Include faiss' cpu ivf pq algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT "Include cuVS ivf flat algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ "Include cuVS ivf pq algorithm in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_CAGRA "Include cuVS CAGRA in benchmark" ON)
option(CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE "Include cuVS brute force knn in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB "Include cuVS CAGRA with HNSW search in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_HNSWLIB "Include hnsw algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_USE_GGNN "Include ggnn algorithm in benchmark" OFF)
option(CUVS_ANN_BENCH_SINGLE_EXE
       "Make a single executable with benchmark as shared library modules" OFF
)

# ##################################################################################################
# * Process options ----------------------------------------------------------

find_package(Threads REQUIRED)

if(BUILD_CPU_ONLY)
  set(CUVS_FAISS_ENABLE_GPU OFF)
  set(CUVS_ANN_BENCH_USE_FAISS_GPU_FLAT OFF)
  set(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT OFF)
  set(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_PQ OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_CAGRA OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE OFF)
  set(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB OFF)
  set(CUVS_ANN_BENCH_USE_GGNN OFF)
else()
  set(CUVS_FAISS_ENABLE_GPU ON)
endif()

set(CUVS_ANN_BENCH_USE_FAISS OFF)
if(CUVS_ANN_BENCH_USE_FAISS_GPU_FLAT
   OR CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_PQ
   OR CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT
   OR CUVS_ANN_BENCH_USE_FAISS_CPU_FLAT
   OR CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_PQ
   OR CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_FLAT
)
  set(CUVS_ANN_BENCH_USE_FAISS ON)
  set(CUVS_USE_FAISS_STATIC ON)
endif()

set(CUVS_ANN_BENCH_USE_CUVS OFF)
if(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ
   OR CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE
   OR CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT
   OR CUVS_ANN_BENCH_USE_CUVS_CAGRA
   OR CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB
)
  set(CUVS_ANN_BENCH_USE_CUVS ON)
endif()

# ##################################################################################################
# * Fetch requirements -------------------------------------------------------------

if(CUVS_ANN_BENCH_USE_HNSWLIB OR CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB)
  include(cmake/thirdparty/get_hnswlib.cmake)
endif()

include(cmake/thirdparty/get_nlohmann_json.cmake)

if(CUVS_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_ggnn.cmake)
endif()

if(CUVS_ANN_BENCH_USE_FAISS)
  # We need to ensure that faiss has all the conda information. So we currently use the very ugly
  # hammer of `link_libraries` to ensure that all targets in this directory and the faiss directory
  # will have the conda includes/link dirs
  link_libraries($<TARGET_NAME_IF_EXISTS:conda_env>)
  include(cmake/thirdparty/get_faiss.cmake)
endif()

# ##################################################################################################
# * Enable NVTX if available

# Note: ANN_BENCH wrappers have extra NVTX code not related to raft::nvtx.They track gbench
# benchmark cases and iterations. This is to make limited NVTX available to all algos, not just
# raft/cuVS.
if(TARGET CUDA::nvtx3)
  set(_CMAKE_REQUIRED_INCLUDES_ORIG ${CMAKE_REQUIRED_INCLUDES})
  get_target_property(CMAKE_REQUIRED_INCLUDES CUDA::nvtx3 INTERFACE_INCLUDE_DIRECTORIES)
  unset(NVTX3_HEADERS_FOUND CACHE)
  # Check the headers explicitly to make sure the cpu-only build succeeds
  CHECK_INCLUDE_FILE_CXX(nvtx3/nvToolsExt.h NVTX3_HEADERS_FOUND)
  set(CMAKE_REQUIRED_INCLUDES ${_CMAKE_REQUIRED_INCLUDES_ORIG})
endif()

# ##################################################################################################
# * Configure tests function-------------------------------------------------------------

function(ConfigureAnnBench)

  set(oneValueArgs NAME)
  set(multiValueArgs PATH LINKS CXXFLAGS)

  if(NOT BUILD_CPU_ONLY)
    set(GPU_BUILD ON)
  endif()

  cmake_parse_arguments(
    ConfigureAnnBench "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN}
  )

  set(BENCH_NAME ${ConfigureAnnBench_NAME}_ANN_BENCH)

  if(CUVS_ANN_BENCH_SINGLE_EXE)
    add_library(${BENCH_NAME} SHARED ${ConfigureAnnBench_PATH})
    string(TOLOWER ${BENCH_NAME} BENCH_LIB_NAME)
    set_target_properties(${BENCH_NAME} PROPERTIES OUTPUT_NAME ${BENCH_LIB_NAME})
    add_dependencies(${BENCH_NAME} ANN_BENCH)
  else()
    add_executable(${BENCH_NAME} ${ConfigureAnnBench_PATH})
    target_compile_definitions(
      ${BENCH_NAME} PRIVATE ANN_BENCH_BUILD_MAIN
                            $<$<BOOL:${NVTX3_HEADERS_FOUND}>:ANN_BENCH_NVTX3_HEADERS_FOUND>
    )
    target_link_libraries(
      ${BENCH_NAME} PRIVATE benchmark::benchmark $<$<BOOL:${NVTX3_HEADERS_FOUND}>:CUDA::nvtx3>
    )
  endif()

  target_link_libraries(
    ${BENCH_NAME}
    PRIVATE ${ConfigureAnnBench_LINKS}
            nlohmann_json::nlohmann_json
            Threads::Threads
            $<$<BOOL:${GPU_BUILD}>:CUDA::cudart_static>
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
            -static-libgcc
            -static-libstdc++
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

  set(${ConfigureAnnBench_CXXFLAGS} ${CUVS_CXX_FLAGS} ${ConfigureAnnBench_CXXFLAGS})

  target_compile_options(
    ${BENCH_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${ConfigureAnnBench_CXXFLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
  )

  if(CUVS_ANN_BENCH_USE_${ConfigureAnnBench_NAME})
    target_compile_definitions(
      ${BENCH_NAME}
      PUBLIC
        CUVS_ANN_BENCH_USE_${ConfigureAnnBench_NAME}=CUVS_ANN_BENCH_USE_${ConfigureAnnBench_NAME}
    )
  endif()

  target_include_directories(
    ${BENCH_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CUVS_SOURCE_DIR}/include>"
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

if(CUVS_ANN_BENCH_USE_HNSWLIB)
  ConfigureAnnBench(
    NAME HNSWLIB PATH bench/ann/src/hnswlib/hnswlib_benchmark.cpp LINKS hnswlib::hnswlib
  )

endif()

if(CUVS_ANN_BENCH_USE_CUVS_IVF_PQ)
  ConfigureAnnBench(
    NAME
    CUVS_IVF_PQ
    PATH
    bench/ann/src/cuvs/cuvs_benchmark.cu
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_IVF_PQ}>:bench/ann/src/cuvs/cuvs_ivf_pq.cu>
    LINKS
    cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT)
  ConfigureAnnBench(
    NAME
    CUVS_IVF_FLAT
    PATH
    bench/ann/src/cuvs/cuvs_benchmark.cu
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_IVF_FLAT}>:bench/ann/src/cuvs/cuvs_ivf_flat.cu>
    LINKS
    cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_BRUTE_FORCE)
  ConfigureAnnBench(
    NAME CUVS_BRUTE_FORCE PATH bench/ann/src/cuvs/cuvs_benchmark.cu LINKS cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_CAGRA)
  ConfigureAnnBench(
    NAME
    CUVS_CAGRA
    PATH
    bench/ann/src/cuvs/cuvs_benchmark.cu
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_CAGRA}>:bench/ann/src/cuvs/cuvs_cagra_float.cu>
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_CAGRA}>:bench/ann/src/cuvs/cuvs_cagra_half.cu>
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_CAGRA}>:bench/ann/src/cuvs/cuvs_cagra_int8_t.cu>
    $<$<BOOL:${CUVS_ANN_BENCH_USE_CUVS_CAGRA}>:bench/ann/src/cuvs/cuvs_cagra_uint8_t.cu>
    LINKS
    cuvs
  )
endif()

if(CUVS_ANN_BENCH_USE_CUVS_CAGRA_HNSWLIB)
  ConfigureAnnBench(
    NAME CUVS_CAGRA_HNSWLIB PATH bench/ann/src/cuvs/cuvs_cagra_hnswlib.cu LINKS cuvs
    hnswlib::hnswlib
  )
endif()

set(CUVS_FAISS_TARGETS faiss::faiss)
if(TARGET faiss::faiss_avx2)
  set(CUVS_FAISS_TARGETS faiss::faiss_avx2)
endif()

message("CUVS_FAISS_TARGETS: ${CUVS_FAISS_TARGETS}")
message("CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
if(CUVS_ANN_BENCH_USE_FAISS_CPU_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_FLAT PATH bench/ann/src/faiss/faiss_cpu_benchmark.cpp LINKS
    ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_FLAT)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_FLAT PATH bench/ann/src/faiss/faiss_cpu_benchmark.cpp LINKS
    ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_CPU_IVF_PQ)
  ConfigureAnnBench(
    NAME FAISS_CPU_IVF_PQ PATH bench/ann/src/faiss/faiss_cpu_benchmark.cpp LINKS
    ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_FLAT)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_FLAT PATH bench/ann/src/faiss/faiss_gpu_benchmark.cu LINKS
    ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_IVF_PQ)
  ConfigureAnnBench(
    NAME FAISS_GPU_IVF_PQ PATH bench/ann/src/faiss/faiss_gpu_benchmark.cu LINKS
    ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_FAISS_GPU_FLAT)
  ConfigureAnnBench(
    NAME FAISS_GPU_FLAT PATH bench/ann/src/faiss/faiss_gpu_benchmark.cu LINKS ${CUVS_FAISS_TARGETS}
  )
endif()

if(CUVS_ANN_BENCH_USE_GGNN)
  include(cmake/thirdparty/get_glog.cmake)
  ConfigureAnnBench(
    NAME GGNN PATH bench/ann/src/ggnn/ggnn_benchmark.cu LINKS glog::glog ggnn::ggnn CUDA::curand
  )
endif()

# ##################################################################################################
# * Dynamically-loading ANN_BENCH executable -------------------------------------------------------
if(CUVS_ANN_BENCH_SINGLE_EXE)
  add_executable(ANN_BENCH bench/ann/src/common/benchmark.cpp)

  # Build and link static version of the GBench to keep ANN_BENCH self-contained.
  get_target_property(TMP_PROP benchmark::benchmark SOURCES)
  add_library(benchmark_static STATIC ${TMP_PROP})
  get_target_property(TMP_PROP benchmark::benchmark INCLUDE_DIRECTORIES)
  target_include_directories(benchmark_static PUBLIC ${TMP_PROP})
  get_target_property(TMP_PROP benchmark::benchmark LINK_LIBRARIES)
  target_link_libraries(benchmark_static PUBLIC ${TMP_PROP})

  target_include_directories(ANN_BENCH PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(
    ANN_BENCH
    PRIVATE raft::raft
            nlohmann_json::nlohmann_json
            benchmark_static
            dl
            -static-libgcc
            fmt::fmt-header-only
            spdlog::spdlog_header_only
            -static-libstdc++
            $<$<BOOL:${NVTX3_HEADERS_FOUND}>:CUDA::nvtx3>
  )
  set_target_properties(
    ANN_BENCH
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
  target_compile_definitions(
    ANN_BENCH
    PRIVATE
      $<$<BOOL:${CUDAToolkit_FOUND}>:ANN_BENCH_LINK_CUDART="libcudart.so.${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}.${CUDAToolkit_VERSION_PATCH}">
      $<$<BOOL:${NVTX3_HEADERS_FOUND}>:ANN_BENCH_NVTX3_HEADERS_FOUND>
  )

  target_link_options(ANN_BENCH PRIVATE -export-dynamic)

  install(
    TARGETS ANN_BENCH
    COMPONENT ann_bench
    DESTINATION bin/ann
    EXCLUDE_FROM_ALL
  )
endif()
