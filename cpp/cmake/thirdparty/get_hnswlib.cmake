#=============================================================================
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

function(find_and_configure_hnswlib)
  message(STATUS "Finding or building hnswlib")
  set(oneValueArgs)

  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/hnswlib_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_info.cmake")
  rapids_cpm_package_info(hnswlib
    VERSION_VAR version
    FIND_VAR find_args
    CPM_VAR cpm_args
    TO_INSTALL_VAR to_install
    BUILD_EXPORT_SET cuvs-exports
    INSTALL_EXPORT_SET cuvs-exports
  )

  rapids_cpm_find(
    hnswlib ${version} ${find_args}
    GLOBAL_TARGETS hnswlib hnswlib::hnswlib
    CPM_ARGS ${cpm_args}
    DOWNLOAD_ONLY ON
  )

  include("${rapids-cmake-dir}/cpm/detail/display_patch_status.cmake")
  rapids_cpm_display_patch_status(hnswlib)

  if(NOT TARGET hnswlib::hnswlib)
    add_library(hnswlib INTERFACE )
    add_library(hnswlib::hnswlib ALIAS hnswlib)
    target_include_directories(hnswlib INTERFACE
     "$<BUILD_INTERFACE:${hnswlib_SOURCE_DIR}>"
     "$<INSTALL_INTERFACE:include>")
  endif()

  if(hnswlib_ADDED)
    # write build export rules
    install(TARGETS hnswlib EXPORT hnswlib-exports)
    if(to_install)
      install(DIRECTORY "${hnswlib_SOURCE_DIR}/hnswlib/" DESTINATION include/hnswlib)

      # write install export rules
      rapids_export(
        INSTALL hnswlib
        VERSION ${version}
        EXPORT_SET hnswlib-exports
        GLOBAL_TARGETS hnswlib
        NAMESPACE hnswlib::)
    endif()

    rapids_export(
      BUILD hnswlib
      VERSION ${version}
      EXPORT_SET hnswlib-exports
      GLOBAL_TARGETS hnswlib
      NAMESPACE hnswlib::)

    # When using cuVS from the build dir, ensure hnswlib is also found in cuVS' build dir. This
    # line adds `set(hnswlib_ROOT "${CMAKE_CURRENT_LIST_DIR}")` to build/cuvs-dependencies.cmake
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      BUILD hnswlib [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cuvs-exports
    )
  endif()
endfunction()

find_and_configure_hnswlib()
