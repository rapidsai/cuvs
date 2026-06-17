# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

# Build-time cuTile cubin targets. Maps to cuvs CUDA 13 -real library arches (75-real omitted).
set(CUTILE_VECTOR_ADD_GPU_CODES sm_80 sm_86 sm_90 sm_100 sm_120)

function(generate_cutile_vector_add_cubins output_include_dir_var)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  find_package(CUDAToolkit REQUIRED)

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
        "cuda.tile (cuTile Python) is required to build CUTILE_VECTOR_ADD_TEST. "
        "Install it in the active Python environment, e.g. pip install cuda-tile[tileiras]."
    )
  endif()

  set(_cutile_source_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  set(_cutile_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/cutile_generated")
  file(MAKE_DIRECTORY "${_cutile_binary_dir}")

  set(_symbol_header "${_cutile_binary_dir}/vector_add_kernel_symbol.h")
  set(_first_gpu_code TRUE)

  foreach(_gpu_code IN LISTS CUTILE_VECTOR_ADD_GPU_CODES)
    set(_cubin_file "${_cutile_binary_dir}/vector_add_${_gpu_code}.cubin")
    set(_cubin_header "${_cutile_binary_dir}/vector_add_${_gpu_code}_cubin.h")

    if(_first_gpu_code)
      set(_symbol_arg --symbol-header "${_symbol_header}")
      set(_cubin_outputs "${_cubin_file}" "${_symbol_header}")
      set(_first_gpu_code FALSE)
    else()
      set(_symbol_arg)
      set(_cubin_outputs "${_cubin_file}")
    endif()

    add_custom_command(
      OUTPUT ${_cubin_outputs}
      COMMAND
        "${Python3_EXECUTABLE}" "${_cutile_source_dir}/export_vector_add_cubin.py"
        "${_cubin_file}" --gpu-code "${_gpu_code}" ${_symbol_arg}
      DEPENDS "${_cutile_source_dir}/export_vector_add_cubin.py"
              "${_cutile_source_dir}/vector_add_kernel.py"
      COMMENT "Exporting cuTile vector_add cubin for ${_gpu_code}"
      VERBATIM
    )

    add_custom_command(
      OUTPUT "${_cubin_header}"
      COMMAND "${CUTILE_BIN2C}" --const --name "vector_add_${_gpu_code}_cubin" --static
              "${_cubin_file}" > "${_cubin_header}"
      DEPENDS "${_cubin_file}"
      COMMENT "Embedding vector_add ${_gpu_code} cubin via bin2c"
      VERBATIM
    )

    list(APPEND _generated_headers "${_cubin_header}")
  endforeach()

  add_custom_target(
    cutile_vector_add_cubins
    DEPENDS "${_symbol_header}" ${_generated_headers}
  )

  set(${output_include_dir_var}
      "${_cutile_binary_dir}"
      PARENT_SCOPE
  )
endfunction()
