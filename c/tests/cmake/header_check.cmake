# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
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

# Build a list of all headers in `c/include/cuvs`

function(cuvs_c_add_header_check project_root binding_header COMPONENT_PLACEHOLDER install_set)
  file(
    GLOB_RECURSE all_headers_to_match
    RELATIVE "${project_root}/include/"
    "${project_root}/include/*.h"
  )

  set(template_contents
      [=[
  set(all_headers_to_match @all_headers_to_match@)
  set(binding_header_name @binding_header@)
  set(binary_dir @CMAKE_CURRENT_BINARY_DIR@)
  set(src_dir @CMAKE_SOURCE_DIR@)

  function(check_binding_header mode header_list_var)

    if(mode STREQUAL BUILD)
      set(path "${src_dir}/include/${binding_header_name}")
    else()
      # Walk up the binary dir till we
      set(path "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include/${binding_header_name}")
    endif()

    if(EXISTS "${path}")
      file(READ "${path}" binding_header_contents)
      string(REPLACE "\n" "" binding_header_contents "${binding_header_contents}")
      foreach(entry ${${header_list_var}})
        if(NOT entry STREQUAL binding_header_name)
          string(FIND "${binding_header_contents}" "<${entry}>" contains)
          if(contains STREQUAL "-1")
            message(FATAL_ERROR "include \"${entry}\" not found in contents of ${binding_header_name}.")
          endif()
        endif()
      endforeach()
    else()
      message(FATAL_ERROR "check_binding_header failed to find ${binding_header_name} on disk.")
    endif()
  endfunction()

  function(check_installed_headers )
    set(path "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include/")
    file(GLOB_RECURSE installed_headers RELATIVE "${path}/" "${path}/cuvs/*.h")
    check_binding_header(INSTALL installed_headers)
  endfunction()

  if(CMAKE_CURRENT_LIST_DIR STREQUAL binary_dir)
    # Build directory checks
    #
    # 1. Check that binding header content includes all headers
    #    that we cached
    #
    check_binding_header(BUILD all_headers_to_match)
  else()
    # Install directory checks
    #
    # 1. Check that all headers we have in our cached list are installed
    # 2. Check that binding header content includes all headers
    #    that have have installed
    check_binding_header(INSTALL all_headers_to_match)
    check_installed_headers()
  endif()

]=]
  )

  set(output_path "${CMAKE_CURRENT_BINARY_DIR}/cuvs_header_check.cmake")
  file(CONFIGURE OUTPUT "${output_path}" CONTENT "${template_contents}" @ONLY)

  # Most likely need to roll all this into a CMake
  add_test(NAME cuvs_c_verify_install_headers COMMAND ${CMAKE_COMMAND} "-P=${output_path}")
  install(
    FILES "${output_path}"
    COMPONENT testing
    DESTINATION "."
    EXCLUDE_FROM_ALL
  )
  set_property(
    TARGET rapids_test_install_${install_set}
    APPEND
    PROPERTY "TESTS_TO_RUN" "cuvs_c_verify_install_headers"
  )
endfunction()
