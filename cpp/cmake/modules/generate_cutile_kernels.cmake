# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/compute_matrix_product.cmake)

function(generate_cutile_kernels_stub)
  set(CUVS_CUTILE_ENABLED 0 PARENT_SCOPE)
endfunction()

function(_cutile_fragment_tag_header_files output_var)
  set(${output_var} "")
  foreach(_header IN LISTS ARGN)
    if(NOT _header MATCHES "^(\".*\"|<.*>)$")
      set(_header "\"${_header}\"")
    endif()
    string(APPEND ${output_var} "#include ${_header}\n")
  endforeach()
  set(${output_var}
      "${${output_var}}"
      PARENT_SCOPE
  )
endfunction()

function(_cutile_kernels_setup)
  set(options)
  set(one_value MATRIX_JSON_FILE OUTPUT_DIRECTORY)
  set(multi_value)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_package(CUDAToolkit REQUIRED)

  if(CUDAToolkit_VERSION VERSION_LESS 13.0)
    message(
      STATUS
        "cuTile embedded kernels require CUDA 13.0+; skipping cuTile generation (found ${CUDAToolkit_VERSION})."
    )
    set(_CUTILE_SETUP_OK
        FALSE
        PARENT_SCOPE
    )
    return()
  endif()

  find_program(
    CUTILE_BIN2C
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR}
    REQUIRED
  )

  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import cuda.tile"
    RESULT_VARIABLE _cutile_import_result
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(NOT _cutile_import_result EQUAL 0)
    message(
      FATAL_ERROR
        "cuda.tile (cuTile Python) is required to build cuTile embedded kernels. "
        "Install it in the active Python environment, e.g. pip install cuda-tile[tileiras]."
    )
  endif()

  set_property(
    DIRECTORY
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_CUTILE_MATRIX_JSON_FILE}"
    APPEND
  )

  file(MAKE_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}")

  set(_CUTILE_SETUP_OK
      TRUE
      PARENT_SCOPE
  )
endfunction()

function(process_cutile_cubin_matrix_entry source_list_var)
  set(options)
  set(one_value
      KERNEL_DIR
      KERNEL_BASENAME
      KERNEL_PYTHON
      EXPORT_SCRIPT
      OUTPUT_DIRECTORY
      FRAGMENT_TAG_FORMAT
      MATRIX_JSON_ENTRY
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_CUTILE_MATRIX_JSON_ENTRY}")
  _cutile_fragment_tag_header_files(
    fragment_tag_header_files ${_CUTILE_FRAGMENT_TAG_HEADER_FILES}
  )

  string(CONFIGURE "${_CUTILE_FRAGMENT_TAG_FORMAT}" fragment_tag @ONLY)

  set(_artifact_basename "${_CUTILE_KERNEL_BASENAME}_${data_type}_${gpu_code}")
  set(_cubin_file "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_basename}.cubin")
  set(_cubin_header "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_basename}_cubin.h")
  set(_cubin_cpp "${_CUTILE_OUTPUT_DIRECTORY}/${_artifact_basename}_cubin.cpp")
  set(cubin_header_file "${_artifact_basename}_cubin.h")

  add_custom_command(
    OUTPUT "${_cubin_file}"
    COMMAND
      "${Python3_EXECUTABLE}" "${_CUTILE_KERNEL_DIR}/${_CUTILE_EXPORT_SCRIPT}" "${_cubin_file}"
      --format cubin --data-type "${data_type}" --gpu-code "${gpu_code}"
    DEPENDS "${_CUTILE_KERNEL_DIR}/${_CUTILE_EXPORT_SCRIPT}"
            "${_CUTILE_KERNEL_DIR}/${_CUTILE_KERNEL_PYTHON}"
    COMMENT "Exporting cuTile ${_CUTILE_KERNEL_BASENAME} cubin ${data_type} ${gpu_code}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${_cubin_header}"
    COMMAND "${CUTILE_BIN2C}" --const --name embedded_cubin --static "${_cubin_file}"
            > "${_cubin_header}"
    DEPENDS "${_cubin_file}"
    VERBATIM
  )

  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/register_cubin.cpp.in" "${_cubin_cpp}" @ONLY
  )
  list(APPEND ${source_list_var} "${_cubin_header}" "${_cubin_cpp}")
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()

function(process_cutile_tileir_matrix_entry source_list_var)
  set(options)
  set(one_value
      KERNEL_DIR
      KERNEL_BASENAME
      KERNEL_PYTHON
      EXPORT_SCRIPT
      OUTPUT_DIRECTORY
      FRAGMENT_TAG_FORMAT
      MATRIX_JSON_ENTRY
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  populate_matrix_variables("${_CUTILE_MATRIX_JSON_ENTRY}")
  _cutile_fragment_tag_header_files(
    fragment_tag_header_files ${_CUTILE_FRAGMENT_TAG_HEADER_FILES}
  )

  string(CONFIGURE "${_CUTILE_FRAGMENT_TAG_FORMAT}" fragment_tag @ONLY)
  set(_tileir_file "${_CUTILE_OUTPUT_DIRECTORY}/${_CUTILE_KERNEL_BASENAME}_${data_type}.tilebc")
  set(_tileir_header "${_CUTILE_OUTPUT_DIRECTORY}/${_CUTILE_KERNEL_BASENAME}_${data_type}_tileir.h")
  set(_tileir_cpp "${_CUTILE_OUTPUT_DIRECTORY}/${_CUTILE_KERNEL_BASENAME}_${data_type}_tileir.cpp")
  set(tileir_header_file "${_CUTILE_KERNEL_BASENAME}_${data_type}_tileir.h")

  add_custom_command(
    OUTPUT "${_tileir_file}"
    COMMAND
      "${Python3_EXECUTABLE}" "${_CUTILE_KERNEL_DIR}/${_CUTILE_EXPORT_SCRIPT}" "${_tileir_file}"
      --format tileir_bytecode --data-type "${data_type}" --gpu-code "${export_gpu_code}"
      --bytecode-version "${bytecode_version}"
    DEPENDS "${_CUTILE_KERNEL_DIR}/${_CUTILE_EXPORT_SCRIPT}"
            "${_CUTILE_KERNEL_DIR}/${_CUTILE_KERNEL_PYTHON}"
    COMMENT "Exporting cuTile ${_CUTILE_KERNEL_BASENAME} TileIR bytecode ${data_type}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT "${_tileir_header}"
    COMMAND "${CUTILE_BIN2C}" --const --name embedded_tileir --static "${_tileir_file}"
            > "${_tileir_header}"
    DEPENDS "${_tileir_file}"
    VERBATIM
  )

  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/register_tileir.cpp.in" "${_tileir_cpp}" @ONLY
  )
  list(APPEND ${source_list_var} "${_tileir_header}" "${_tileir_cpp}")
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()

function(generate_cutile_cubin_kernels source_list_var)
  set(options)
  set(one_value
      KERNEL_DIR
      KERNEL_BASENAME
      KERNEL_PYTHON
      EXPORT_SCRIPT
      OUTPUT_DIRECTORY
      MATRIX_JSON_FILE
      FRAGMENT_TAG_FORMAT
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT _CUTILE_KERNEL_BASENAME)
    message(FATAL_ERROR "generate_cutile_cubin_kernels: KERNEL_BASENAME is required")
  endif()
  if(NOT _CUTILE_KERNEL_PYTHON)
    set(_CUTILE_KERNEL_PYTHON "fused_1nn_kernel.py")
  endif()

  _cutile_kernels_setup(
    MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}"
    OUTPUT_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}"
  )
  if(NOT _CUTILE_SETUP_OK)
    generate_cutile_kernels_stub()
    set(${source_list_var}
        ""
        PARENT_SCOPE
    )
    return()
  endif()

  compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}")

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_cutile_cubin_matrix_entry(
      "${source_list_var}"
      KERNEL_DIR "${_CUTILE_KERNEL_DIR}"
      KERNEL_BASENAME "${_CUTILE_KERNEL_BASENAME}"
      KERNEL_PYTHON "${_CUTILE_KERNEL_PYTHON}"
      EXPORT_SCRIPT "${_CUTILE_EXPORT_SCRIPT}"
      OUTPUT_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}"
      FRAGMENT_TAG_FORMAT "${_CUTILE_FRAGMENT_TAG_FORMAT}"
      FRAGMENT_TAG_HEADER_FILES ${_CUTILE_FRAGMENT_TAG_HEADER_FILES}
      MATRIX_JSON_ENTRY "${matrix_json_entry}"
    )
  endforeach()

  set(CUVS_CUTILE_ENABLED 1 PARENT_SCOPE)
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()

function(generate_cutile_tileir_kernels source_list_var)
  set(options)
  set(one_value
      KERNEL_DIR
      KERNEL_BASENAME
      KERNEL_PYTHON
      EXPORT_SCRIPT
      OUTPUT_DIRECTORY
      MATRIX_JSON_FILE
      FRAGMENT_TAG_FORMAT
  )
  set(multi_value FRAGMENT_TAG_HEADER_FILES)
  cmake_parse_arguments(_CUTILE "${options}" "${one_value}" "${multi_value}" ${ARGN})

  if(NOT _CUTILE_KERNEL_BASENAME)
    message(FATAL_ERROR "generate_cutile_tileir_kernels: KERNEL_BASENAME is required")
  endif()
  if(NOT _CUTILE_KERNEL_PYTHON)
    set(_CUTILE_KERNEL_PYTHON "fused_1nn_kernel.py")
  endif()

  _cutile_kernels_setup(
    MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}"
    OUTPUT_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}"
  )
  if(NOT _CUTILE_SETUP_OK)
    generate_cutile_kernels_stub()
    return()
  endif()

  compute_matrix_product(matrix_product MATRIX_JSON_FILE "${_CUTILE_MATRIX_JSON_FILE}")

  string(JSON len LENGTH "${matrix_product}")
  math(EXPR last "${len} - 1")

  # cmake-lint: disable=C0103,E1120
  foreach(i RANGE "${last}")
    string(JSON matrix_json_entry GET "${matrix_product}" "${i}")
    process_cutile_tileir_matrix_entry(
      "${source_list_var}"
      KERNEL_DIR "${_CUTILE_KERNEL_DIR}"
      KERNEL_BASENAME "${_CUTILE_KERNEL_BASENAME}"
      KERNEL_PYTHON "${_CUTILE_KERNEL_PYTHON}"
      EXPORT_SCRIPT "${_CUTILE_EXPORT_SCRIPT}"
      OUTPUT_DIRECTORY "${_CUTILE_OUTPUT_DIRECTORY}"
      FRAGMENT_TAG_FORMAT "${_CUTILE_FRAGMENT_TAG_FORMAT}"
      FRAGMENT_TAG_HEADER_FILES ${_CUTILE_FRAGMENT_TAG_HEADER_FILES}
      MATRIX_JSON_ENTRY "${matrix_json_entry}"
    )
  endforeach()

  set(CUVS_CUTILE_ENABLED 1 PARENT_SCOPE)
  set(${source_list_var}
      "${${source_list_var}}"
      PARENT_SCOPE
  )
endfunction()
