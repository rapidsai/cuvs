#=============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_bang)
  set(oneValueArgs VERSION FORK PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )
  
  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/bang_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(bang version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(bang ${version} patch_command)

  rapids_cpm_find(
    bang ${version}
    GLOBAL_TARGETS bang
    CPM_ARGS
    GIT_REPOSITORY ${repository} 
    SOURCE_SUBDIR BANG_Base
    GIT_TAG ${tag}
    GIT_SHALLOW ${shallow}
  )

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(bang)

  if(NOT TARGET bang::bang)
    target_include_directories(bang INTERFACE "$<BUILD_INTERFACE:${bang_SOURCE_DIR}/BANG_Base>")
    add_library(bang::bang ALIAS bang)
  endif()

  #if(bang_ADDED)
  #  rapids_export(BUILD bang
  #                EXPORT_SET bang-targets
  #                NAMESPACE bang::)
  #endif()

endfunction()

find_and_configure_bang(VERSION  0
        FORK             tarang-jain
        PINNED_TAG       rapids
        )
