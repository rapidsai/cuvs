# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

function(compute_matrix_product output_var)
  set(options)
  set(one_value MATRIX_JSON_FILE MATRIX_JSON_STRING)
  set(multi_value)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(_JIT_LTO_MATRIX_JSON_FILE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compute_matrix_product.py"
              "${_JIT_LTO_MATRIX_JSON_FILE}" #
      OUTPUT_VARIABLE output COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E echo "${_JIT_LTO_MATRIX_JSON_STRING}"
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/compute_matrix_product.py"
              -
      OUTPUT_VARIABLE output COMMAND_ERROR_IS_FATAL ANY
    )
  endif()

  set(${output_var}
      "${output}"
      PARENT_SCOPE
  )
endfunction()

function(populate_matrix_variables matrix_json_entry)
  string(JSON len LENGTH "${matrix_json_entry}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON key MEMBER "${matrix_json_entry}" "${i}")
    string(JSON value GET "${matrix_json_entry}" "${key}")
    set(${key}
        "${value}"
        PARENT_SCOPE
    )
  endforeach()
endfunction()

function(add_jit_lto_kernel kernel_target)
  set(options)
  set(one_value KERNEL_FILE EMBEDDED_HEADER_FILE)
  set(multi_value LINK_LIBRARIES)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  add_library(${kernel_target} OBJECT EXCLUDE_FROM_ALL "${_JIT_LTO_KERNEL_FILE}")
  # Do not modify these properties, options, and libraries. Usage requirements (including CUDA
  # version, etc.) should be propagated to the kernel targets via INTERFACE libraries passed in
  # through the LINK_LIBRARIES argument.
  target_link_libraries(${kernel_target} PRIVATE ${_JIT_LTO_LINK_LIBRARIES})
  target_compile_options(${kernel_target} PRIVATE -Xfatbin=--compress-all --compress-mode=size)
  set_target_properties(
    ${kernel_target}
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
               CUDA_FATBIN_COMPILATION ON
               POSITION_INDEPENDENT_CODE ON
               INTERPROCEDURAL_OPTIMIZATION ON
  )

  add_custom_command(
    OUTPUT "${_JIT_LTO_EMBEDDED_HEADER_FILE}"
    COMMAND "${bin_to_c}" --const --name embedded_fatbin --static $<TARGET_OBJECTS:${kernel_target}>
            > "${_JIT_LTO_EMBEDDED_HEADER_FILE}"
    DEPENDS $<TARGET_OBJECTS:${kernel_target}>
  )
endfunction()

function(process_matrix_entry source_list_var)
  set(options)
  set(one_value NAME_FORMAT KERNEL_INPUT_FILE EMBEDDED_INPUT_FILE OUTPUT_DIRECTORY
                MATRIX_JSON_ENTRY
  )
  set(multi_value KERNEL_LINK_LIBRARIES)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_JIT_LTO_MATRIX_JSON_ENTRY}")
  string(CONFIGURE "${_JIT_LTO_NAME_FORMAT}" kernel_name @ONLY)

  set(kernel_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_kernel.cu")
  set(kernel_target "${kernel_name}_kernel")
  set(embedded_header_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_embedded.h")
  set(embedded_file "${_JIT_LTO_OUTPUT_DIRECTORY}/${kernel_name}_embedded.cpp")
  configure_file("${_JIT_LTO_KERNEL_INPUT_FILE}" "${kernel_file}" @ONLY)
  configure_file("${_JIT_LTO_EMBEDDED_INPUT_FILE}" "${embedded_file}" @ONLY)

  add_jit_lto_kernel(
    ${kernel_target}
    KERNEL_FILE "${kernel_file}"
    EMBEDDED_HEADER_FILE "${embedded_header_file}"
    LINK_LIBRARIES ${_JIT_LTO_KERNEL_LINK_LIBRARIES}
  )
  list(APPEND ${source_list_var} "${embedded_header_file}" "${embedded_file}")
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()

function(generate_jit_lto_kernels source_list_var)
  set(options)
  set(one_value NAME_FORMAT MATRIX_JSON_FILE MATRIX_JSON_STRING KERNEL_INPUT_FILE
                EMBEDDED_INPUT_FILE OUTPUT_DIRECTORY
  )
  set(multi_value KERNEL_LINK_LIBRARIES)

  cmake_parse_arguments(_JIT_LTO "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(CUDAToolkit REQUIRED)
  find_program(
    bin_to_c
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR}
  )
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  if(_JIT_LTO_MATRIX_JSON_FILE)
    set_property(
      DIRECTORY
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${_JIT_LTO_MATRIX_JSON_FILE}"
      APPEND
    )
    compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_JIT_LTO_MATRIX_JSON_FILE}")
  else()
    compute_matrix_product(matrix_product MATRIX_JSON_STRING "${_JIT_LTO_MATRIX_JSON_STRING}")
  endif()

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_matrix_entry(
      "${source_list_var}"
      NAME_FORMAT "${_JIT_LTO_NAME_FORMAT}"
      KERNEL_INPUT_FILE "${_JIT_LTO_KERNEL_INPUT_FILE}"
      EMBEDDED_INPUT_FILE "${_JIT_LTO_EMBEDDED_INPUT_FILE}"
      OUTPUT_DIRECTORY "${_JIT_LTO_OUTPUT_DIRECTORY}"
      MATRIX_JSON_ENTRY "${matrix_json_entry}"
      KERNEL_LINK_LIBRARIES ${_JIT_LTO_KERNEL_LINK_LIBRARIES}
    )
  endforeach()

  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
