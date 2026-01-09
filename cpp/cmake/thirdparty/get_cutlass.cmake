# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
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

  set(CUDART_LIBRARY "${CUDA_cudart_static_LIBRARY}" CACHE FILEPATH "fixing cutlass cmake code" FORCE)

  include("${rapids-cmake-dir}/cpm/package_override.cmake")
  rapids_cpm_package_override("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches/cutlass_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(cutlass
    VERSION_VAR version
    FIND_VAR find_args
    CPM_VAR cpm_args
  )

  rapids_cpm_find(
    NvidiaCutlass ${version} ${find_args}
    GLOBAL_TARGETS nvidia::cutlass::cutlass
    CPM_ARGS ${cpm_args}
    OPTIONS "CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_DIR}"
  )

  if(TARGET CUTLASS AND NOT TARGET nvidia::cutlass::cutlass)
    add_library(nvidia::cutlass::cutlass ALIAS CUTLASS)
  endif()

  # export cutlass so projects that depend on cuVS can find it
  if(NvidiaCutlass_ADDED)
    rapids_export(
      BUILD NvidiaCutlass
      EXPORT_SET NvidiaCutlass
      GLOBAL_TARGETS nvidia::cutlass::cutlass
      NAMESPACE nvidia::cutlass::
    )
  endif()

  rapids_export_package(
      BUILD NvidiaCutlass cuvs-static-exports GLOBAL_TARGETS nvidia::cutlass::cutlass
  )

  # Tell cmake where it can find the generated NvidiaCutlass-config.cmake we wrote.
  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(
          BUILD NvidiaCutlass [=[${CMAKE_CURRENT_LIST_DIR}]=]
          EXPORT_SET cuvs-static-exports
  )

endfunction()

find_and_configure_cutlass()
