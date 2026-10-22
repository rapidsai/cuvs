# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Private binary-size experiment; defaults remain unchanged.
option(CUVS_EXPERIMENT_DEDUP_KERNELS "Compile duplicate CUDA owners together" OFF)
if(CUVS_EXPERIMENT_DEDUP_KERNELS)
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" "${CMAKE_CURRENT_LIST_DIR}/prepare_kernel_owners.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/.." "${CMAKE_CURRENT_BINARY_DIR}"
      "${CMAKE_CURRENT_LIST_DIR}/kernel_owner_groups.json" COMMAND_ERROR_IS_FATAL ANY
  )
  include("${CMAKE_CURRENT_BINARY_DIR}/dedup_owners/owners.cmake")
  get_target_property(_dedup_sources cuvs_objs SOURCES)
  set(_dedup_retained)
  foreach(_dedup_source IN LISTS _dedup_sources)
    get_filename_component(
      _dedup_absolute "${_dedup_source}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    if(NOT _dedup_absolute IN_LIST _dedup_originals)
      list(APPEND _dedup_retained "${_dedup_source}")
    endif()
  endforeach()
  set_property(TARGET cuvs_objs PROPERTY SOURCES "${_dedup_retained};${_dedup_copies}")
  target_sources(cuvs_objs PRIVATE src/detail/kernel_dedup_abi.cpp)
  set_target_properties(cuvs_objs PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
endif()
