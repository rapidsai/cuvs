# =============================================================================
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
include(../rapids_config.cmake)
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
  rapids_cuda_init_architectures(CUVS)
  list(APPEND lang_list "CUDA")
endif()

project(
  CUVS
  VERSION "${RAPIDS_VERSION}"
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
option(BUILD_C_LIBRARY "Build raft C API library" OFF)
option(BUILD_C_TESTS "Build raft C API tests" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_STATIC_RUNTIME "Statically link the CUDA toolkit runtime and libraries" OFF)
option(CUVS_USE_RAFT_STATIC "Build and statically link the RAFT libraries" OFF)
option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(CUVS_NVTX "Enable nvtx markers" OFF)

if((BUILD_TESTS OR BUILD_C_LIBRARY) AND NOT BUILD_CPU_ONLY)

endif()

if(BUILD_CPU_ONLY)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_TESTS OFF)
  set(BUILD_C_LIBRARY OFF)
endif()

if(NOT BUILD_C_LIBRARY)
  set(BUILD_C_TESTS OFF)
endif()

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)
# cmake_dependent_option( CUVS_USE_FAISS_STATIC "Build and statically link the FAISS library for
# nearest neighbors search on GPU" ON CUVS_COMPILE_LIBRARY OFF )

message(VERBOSE "cuVS: Build cuVS unit-tests: ${BUILD_TESTS}")
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
message(VERBOSE "cuVS: Build and statically link RAFT libraries: ${CUVS_USE_RAFT_STATIC}")

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

if(BUILD_C_LIBRARY)
  include(cmake/thirdparty/get_dlpack.cmake)
endif()

if(BUILD_TESTS OR BUILD_C_TESTS)
  include(cmake/thirdparty/get_gtest.cmake)
endif()

include(cmake/thirdparty/get_cutlass.cmake)

# ##################################################################################################
# * cuvs ---------------------------------------------------------------------

add_library(
  cuvs SHARED
  src/neighbors/brute_force_index.cu
  src/neighbors/brute_force.cu
  src/neighbors/cagra_build_float.cpp
  src/neighbors/cagra_build_int8.cpp
  src/neighbors/cagra_build_uint8.cpp
  src/neighbors/cagra_optimize.cpp
  src/neighbors/cagra_search_float.cpp
  src/neighbors/cagra_search_int8.cpp
  src/neighbors/cagra_search_uint8.cpp
  src/neighbors/cagra_serialize_float.cpp
  src/neighbors/cagra_serialize_int8.cpp
  src/neighbors/cagra_serialize_uint8.cpp
  src/neighbors/ivf_flat_index.cpp
  src/neighbors/ivf_flat/ivf_flat_build_float_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_build_int8_t_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_build_uint8_t_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_extend_float_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_extend_int8_t_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_extend_uint8_t_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_search_float_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_search_int8_t_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_search_uint8_t_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_serialize_float_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_serialize_int8_t_int64_t.cpp
  src/neighbors/ivf_flat/ivf_flat_serialize_uint8_t_int64_t.cpp
  src/neighbors/ivf_pq_index.cpp
  src/neighbors/ivf_pq/ivf_pq_build_float_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_build_int8_t_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_build_uint8_t_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_extend_float_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_extend_int8_t_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_extend_uint8_t_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_search_float_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_search_int8_t_int64_t.cpp
  src/neighbors/ivf_pq/ivf_pq_search_uint8_t_int64_t.cpp
  src/neighbors/ivf_pq_serialize.cpp
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
  target_link_libraries(cuvs
        PUBLIC
            rmm::rmm
            $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::raft>
            $<$<NOT:$<BOOL:${CUDA_STATIC_RUNTIME}>>:raft::compiled>
        PRIVATE
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::raft>
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::compiled_static>
            nvidia::cutlass::cutlass
    )
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
  target_link_libraries(cuvs PUBLIC CUDA::nvtx3)
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

target_link_libraries(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:CUDA::nvtx3>)
target_compile_definitions(cuvs::cuvs INTERFACE $<$<BOOL:${CUVS_NVTX}>:NVTX_ENABLED>)

  ]=]
  )
endif()

set_target_properties(
  cuvs
  PROPERTIES BUILD_RPATH                         "\$ORIGIN"
             INSTALL_RPATH                       "\$ORIGIN"
             CXX_STANDARD                        17
             CXX_STANDARD_REQUIRED               ON
             CUDA_STANDARD                       17
             CUDA_STANDARD_REQUIRED              ON
             INTERFACE_POSITION_INDEPENDENT_CODE ON
             POSITION_INDEPENDENT_CODE           ON
)

target_compile_options(
  cuvs PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
               "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
)
# ensure CUDA symbols aren't relocated to the middle of the debug build binaries
target_link_options(cuvs PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")

# ##################################################################################################
# * cuvs_c -------------------------------------------------------------------------------
if(BUILD_C_LIBRARY)
  add_library(
    cuvs_c SHARED
    src/core/c_api.cpp
    src/neighbors/brute_force_c.cpp
    src/neighbors/ivf_flat_c.cpp
    src/neighbors/ivf_pq_c.cpp
    src/neighbors/cagra_c.cpp
  )

  add_library(cuvs::c_api ALIAS cuvs_c)

  set_target_properties(
    cuvs_c
    PROPERTIES BUILD_RPATH                         "\$ORIGIN"
               INSTALL_RPATH                       "\$ORIGIN"
               CXX_STANDARD                        17
               CXX_STANDARD_REQUIRED               ON
               POSITION_INDEPENDENT_CODE           ON
               INTERFACE_POSITION_INDEPENDENT_CODE ON
               EXPORT_NAME                         c_api
  )

  target_compile_options(cuvs_c PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>")

  target_include_directories(
    cuvs_c
    PUBLIC "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>"
    INTERFACE "$<INSTALL_INTERFACE:include>"
  )

  target_link_libraries(cuvs_c
        PUBLIC
            cuvs::cuvs
        PRIVATE
            $<$<BOOL:${CUDA_STATIC_RUNTIME}>:raft::raft>
    )

  # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
  target_link_options(cuvs_c PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
endif()

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

if(BUILD_C_LIBRARY)
  install(
    TARGETS cuvs_c
    DESTINATION ${lib_dir}
    COMPONENT c_api
    EXPORT cuvs-c-exports
  )
endif()

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/cuvs/version_config.hpp
  COMPONENT cuvs
  DESTINATION include/cuvs
)

if(TARGET cuvs_c)
  list(APPEND cuvs_components c_api)
  list(APPEND cuvs_export_sets cuvs-c-exports)
  set(CUVS_C_TARGET cuvs_c)
endif()

# Use `rapids_export` for 22.04 as it will have COMPONENT support
rapids_export(
  INSTALL cuvs
  EXPORT_SET cuvs-exports
  COMPONENTS ${cuvs_components}
  COMPONENTS_EXPORT_SET ${cuvs_export_sets}
  GLOBAL_TARGETS cuvs ${CUVS_C_TARGET}
  NAMESPACE cuvs::
)

# ##################################################################################################
# * build export -------------------------------------------------------------
rapids_export(
  BUILD cuvs
  EXPORT_SET cuvs-exports
  COMPONENTS ${cuvs_components}
  COMPONENTS_EXPORT_SET ${cuvs_export_sets}
  GLOBAL_TARGETS cuvs ${CUVS_C_TARGET}
  NAMESPACE cuvs::
)

# ##################################################################################################
# * build test executable ----------------------------------------------------

if(BUILD_TESTS OR BUILD_C_TESTS)
  include(internal/CMakeLists.txt)
  include(test/CMakeLists.txt)
endif()
