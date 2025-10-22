#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_hnswlib)
  message(STATUS "Finding or building hnswlib")
  set(oneValueArgs)

  include(${rapids-cmake-dir}/cpm/package_override.cmake)
  set(patch_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../patches")
  rapids_cpm_package_override("${patch_dir}/hnswlib_override.json")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(hnswlib version repository tag shallow exclude)

  include("${rapids-cmake-dir}/cpm/detail/generate_patch_command.cmake")
  rapids_cpm_generate_patch_command(hnswlib ${version} patch_command build_patch_only)

  rapids_cpm_find(
    hnswlib ${version} ${build_patch_only}
    GLOBAL_TARGETS hnswlib hnswlib::hnswlib
    CPM_ARGS
    GIT_REPOSITORY ${repository}
    GIT_TAG ${tag}
    GIT_SHALLOW ${shallow} ${patch_command}
    EXCLUDE_FROM_ALL ${exclude}
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
    if(NOT exclude)
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

    include("${rapids-cmake-dir}/export/package.cmake")
    rapids_export_package(INSTALL hnswlib cuvs-exports VERSION ${version} GLOBAL_TARGETS hnswlib hnswlib::hnswlib)
    rapids_export_package(BUILD hnswlib cuvs-exports VERSION ${version} GLOBAL_TARGETS hnswlib hnswlib::hnswlib)


    # When using cuVS from the build dir, ensure hnswlib is also found in cuVS' build dir. This
    # line adds `set(hnswlib_ROOT "${CMAKE_CURRENT_LIST_DIR}")` to build/cuvs-dependencies.cmake
    include("${rapids-cmake-dir}/export/find_package_root.cmake")
    rapids_export_find_package_root(
      BUILD hnswlib [=[${CMAKE_CURRENT_LIST_DIR}]=] EXPORT_SET cuvs-exports
    )
  endif()
endfunction()

find_and_configure_hnswlib()
