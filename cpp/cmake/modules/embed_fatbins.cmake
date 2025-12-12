# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include_guard(GLOBAL)

function(embed_fatbins library_name kernel_target)
  find_package(CUDAToolkit REQUIRED)
  find_program(
    bin_to_c
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR}
  )

  set(output_dir ${CMAKE_CURRENT_BINARY_DIR}/${library_name})

  # Create a response file to avoid "argument list too long" errors
  set(objects_response_file ${CMAKE_CURRENT_BINARY_DIR}/embed_fatbins/${library_name}_objects.rsp)

  # Write the objects list to a response file using file(GENERATE) which handles generator
  # expressions
  file(
    GENERATE
    OUTPUT "${objects_response_file}"
    CONTENT "$<JOIN:$<TARGET_OBJECTS:${kernel_target}>,\n>\n"
  )

  # Generate individual headers for each FATBIN object
  add_custom_command(
    OUTPUT "${output_dir}/headers_generated.stamp"
    COMMAND
      ${CMAKE_COMMAND} "-DBIN_TO_C_COMMAND=${bin_to_c}"
      "-DOBJECTS_RESPONSE_FILE=${objects_response_file}" "-DOUTPUT_DIR=${output_dir}"
      "-DSTAMP_FILE=${output_dir}/headers_generated.stamp" -P
      ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/detail/generate_header.cmake
    VERBATIM
    DEPENDS "${objects_response_file}" $<TARGET_OBJECTS:${kernel_target}>
    COMMENT "Converting FATBIN kernels to individual C++ headers"
  )

  # get the sources of `kernel_target` and add them as CUDA sources so we re-compile them to get the
  # inline registration logic
  get_target_property(output_sources ${kernel_target} SOURCES)

  # add those c++ sources to `library_name`
  target_sources(${library_name} PRIVATE "${output_dir}/headers_generated.stamp" ${output_sources})
  target_compile_features(${library_name} PRIVATE cxx_std_20)
  target_include_directories(${library_name} PRIVATE ${output_dir})
endfunction()
