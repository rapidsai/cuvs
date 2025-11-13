# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

function(generate_cuvs_export )
  set(options "CLIB")
  set(one_value "")
  set(multi_value EXPORT_SETS COMPONENTS)
  cmake_parse_arguments(_CUVS_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})


  set(placehold_name "cuvs_placeholder")
  if(_CUVS_RAPIDS_CLIB)
    string(APPEND placehold_name "_c")
  endif()

  # We need a placeholder target in cuvs-exports so that we can generate an export set since all the
  # real targets are conditional
  #
  # tasks: need a second name for when calling from cuvs_c
  # tasks: need to write a query over set(cuvs_comp_names cuvs_shared;cuvs_static;cuvs_cpp_headers;c_api)
  #.       since we need that to be invariant
  if(NOT TARGET cuvs::${placehold_name})
    add_library(${placehold_name} INTERFACE)
    add_library(cuvs::${placehold_name} ALIAS ${placehold_name})
    install(
        TARGETS ${placehold_name}
        EXPORT cuvs-exports
    )
  endif()

  # tasks we need to add in dependencies file inclusion as well

  set(cuvs_final_code_block
  [=[

  if(NOT TARGET cuvs::cuvs_cpp_headers)

    file(GLOB cuvs_component_dep_files LIST_DIRECTORIES FALSE
            "${CMAKE_CURRENT_LIST_DIR}/cuvs-cuvs_cpp_headers*-dependencies.cmake")
    foreach(f IN LISTS  cuvs_component_dep_files)
      include("${f}")
    endforeach()

    file(GLOB cuvs_component_target_files LIST_DIRECTORIES FALSE
                "${CMAKE_CURRENT_LIST_DIR}/cuvs-cuvs_cpp_headers*-targets.cmake")

    foreach(f IN LISTS  cuvs_component_target_files)
      include("${f}")
    endforeach()
  endif()

  if(NOT cuvs_FIND_COMPONENTS)

    file(GLOB cuvs_component_dep_files LIST_DIRECTORIES FALSE
            "${CMAKE_CURRENT_LIST_DIR}/cuvs-cuvs_shared*-dependencies.cmake")
    foreach(f IN LISTS  cuvs_component_dep_files)
      include("${f}")
    endforeach()

    file(GLOB cuvs_component_target_files LIST_DIRECTORIES FALSE
            "${CMAKE_CURRENT_LIST_DIR}/cuvs-cuvs_shared*-targets.cmake")
    foreach(f IN LISTS  cuvs_component_target_files)
      include("${f}")
    endforeach()
  endif()

  foreach(target IN LISTS rapids_namespaced_global_targets)
    if(TARGET ${target})
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
      if(_is_imported AND NOT _already_global)
          set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
    endif()
  endforeach()
  ]=])

  rapids_export(
    INSTALL cuvs
    VERSION "${RAPIDS_VERSION}"
    EXPORT_SET cuvs-exports
    COMPONENTS ${_CUVS_RAPIDS_COMPONENTS}
    COMPONENTS_EXPORT_SET ${_CUVS_RAPIDS_EXPORT_SETS}
    GLOBAL_TARGETS cuvs cuvs_cpp_headers cuvs_static cuvs_c
    NAMESPACE cuvs::
    FINAL_CODE_BLOCK cuvs_final_code_block
  )

  # ################################################################################################
  # * build export -------------------------------------------------------------
  rapids_export(
    BUILD cuvs
    VERSION "${RAPIDS_VERSION}"
    EXPORT_SET cuvs-exports
    COMPONENTS ${_CUVS_RAPIDS_COMPONENTS}
    COMPONENTS_EXPORT_SET ${_CUVS_RAPIDS_EXPORT_SETS}
    GLOBAL_TARGETS cuvs cuvs_cpp_headers cuvs_static cuvs_c
    NAMESPACE cuvs::
    FINAL_CODE_BLOCK cuvs_final_code_block
  )
endfunction()
