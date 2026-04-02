# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

function(process_inst_matrix_entry source_list_var)
  set(options)
  set(one_value NAME_FORMAT INPUT_FILE OUTPUT_DIRECTORY MATRIX_JSON_ENTRY)
  set(multi_value)

  cmake_parse_arguments(_GIM "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_GIM_MATRIX_JSON_ENTRY}")
  string(CONFIGURE "${_GIM_NAME_FORMAT}" inst_name @ONLY)

  set(output_file "${_GIM_OUTPUT_DIRECTORY}/${inst_name}_inst.cu")
  configure_file("${_GIM_INPUT_FILE}" "${output_file}" @ONLY)

  list(APPEND ${source_list_var} "${output_file}")
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()

function(generate_inst_matrix source_list_var)
  set(options)
  set(one_value NAME_FORMAT MATRIX_JSON_FILE MATRIX_JSON_STRING INPUT_FILE OUTPUT_DIRECTORY)
  set(multi_value)

  cmake_parse_arguments(_GIM "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(_GIM_MATRIX_JSON_FILE)
    set_property(
      DIRECTORY
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${_GIM_MATRIX_JSON_FILE}"
      APPEND
    )
    compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_GIM_MATRIX_JSON_FILE}")
  else()
    compute_matrix_product(matrix_product MATRIX_JSON_STRING "${_GIM_MATRIX_JSON_STRING}")
  endif()

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_inst_matrix_entry(
      "${source_list_var}"
      NAME_FORMAT "${_GIM_NAME_FORMAT}"
      INPUT_FILE "${_GIM_INPUT_FILE}"
      OUTPUT_DIRECTORY "${_GIM_OUTPUT_DIRECTORY}"
      MATRIX_JSON_ENTRY "${matrix_json_entry}"
    )
  endforeach()

  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
