#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_diskann)
  set(oneValueArgs VERSION REPOSITORY PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

    include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/diskann_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(diskann version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(diskann ${version} patch_command build_patch_only)

  rapids_cpm_find(diskann ${version} ${build_patch_only}
          GLOBAL_TARGETS diskann
          CPM_ARGS
          OPTIONS
          "PYBIND OFF"
          "UNIT_TEST OFF"
          "RESTAPI OFF"
          "PORTABLE OFF")

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(diskann)

  if(NOT TARGET diskann::diskann)
      target_include_directories(diskann INTERFACE "$<BUILD_INTERFACE:${diskann_SOURCE_DIR}/include>")
      add_library(diskann::diskann ALIAS diskann)
  endif()
endfunction()
find_and_configure_diskann()
