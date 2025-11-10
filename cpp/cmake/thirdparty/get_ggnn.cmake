#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_ggnn)

  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/ggnn_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(ggnn
    VERSION_VAR version
    FIND_VAR find_args
    CPM_VAR cpm_args
  )

  rapids_cpm_find(
    ggnn ${version} ${find_args}
    GLOBAL_TARGETS ggnn::ggnn
    CPM_ARGS ${cpm_args}
    DOWNLOAD_ONLY ON
  )

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(ggnn)

  if(NOT TARGET ggnn::ggnn)
    add_library(ggnn INTERFACE)
    target_include_directories(ggnn INTERFACE "$<BUILD_INTERFACE:${ggnn_SOURCE_DIR}/include>")
    add_library(ggnn::ggnn ALIAS ggnn)
  endif()

endfunction()
find_and_configure_ggnn()
