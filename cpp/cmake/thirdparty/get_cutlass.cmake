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

function(find_and_configure_cutlass)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(CUTLASS_ENABLE_HEADERS_ONLY
      ON
      CACHE BOOL "Enable only the header library"
  )
  set(CUTLASS_NAMESPACE
      "cuvs_cutlass"
      CACHE STRING "Top level namespace of CUTLASS"
  )
  set(CUTLASS_ENABLE_CUBLAS
      OFF
      CACHE BOOL "Disable CUTLASS to build with cuBLAS library."
  )

  if (CUDA_STATIC_RUNTIME)
    set(CUDART_LIBRARY "${CUDA_cudart_static_LIBRARY}" CACHE FILEPATH "fixing cutlass cmake code" FORCE)
  endif()

  include("${rapids-cmake-dir}/cpm/package_override.cmake")
  rapids_cpm_package_override("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches/cutlass_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(cutlass version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(cutlass ${version} patch_command build_patch_only)

  rapids_cpm_find(
    NvidiaCutlass ${version} ${build_patch_only}
    GLOBAL_TARGETS nvidia::cutlass::cutlass
    CPM_ARGS
    GIT_REPOSITORY ${repository}
    GIT_TAG ${tag}
    GIT_SHALLOW ${shallow} ${patch_command}
    OPTIONS "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
  )

  if(TARGET CUTLASS AND NOT TARGET nvidia::cutlass::cutlass)
    add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
  endif()

endfunction()

find_and_configure_cutlass()
