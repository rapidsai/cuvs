# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds Eigen3 and sets any additional necessary environment variables.
function(find_eigen VERSION)
  set(oneValueArgs VERSION)

  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/eigen_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(Eigen3
    VERSION_VAR version
    FIND_VAR find_args
    CPM_VAR cpm_args
  )

  rapids_cpm_find(
    Eigen3 ${version} ${find_args}
    GLOBAL_TARGETS Eigen3 Eigen3::Eigen
    CPM_ARGS {cpm_args}
  )

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(Eigen3)

endfunction()

find_eigen(VERSION 3.4.0)

