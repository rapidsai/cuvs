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

# Generate JIT LTO kernel files at build time using a Python generator script Arguments: kernel_name
# - Name of the kernel type (e.g., "interleaved_scan") generator_script - Path to the Python script
# that generates the kernels
function(generate_jit_lto_kernels kernel_name generator_script)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(OUTPUT_BASE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated_kernels)
  set(GENERATED_CMAKE_FILE ${OUTPUT_BASE_DIR}/${kernel_name}.cmake)

  # Generate the kernels at build time
  add_custom_command(
    OUTPUT ${GENERATED_CMAKE_FILE}
    COMMAND ${Python3_EXECUTABLE} ${generator_script} ${OUTPUT_BASE_DIR} ${kernel_name}
    DEPENDS ${generator_script}
    COMMENT "Generating ${kernel_name} kernel files..."
    VERBATIM
  )

  # Create a custom target that depends on the generated CMake file Use a unique target name based
  # on the kernel name
  set(TARGET_NAME "generate_${kernel_name}_kernels_target")
  add_custom_target(${TARGET_NAME} DEPENDS ${GENERATED_CMAKE_FILE})

  # Only generate if the CMake file doesn't exist
  if(NOT EXISTS ${GENERATED_CMAKE_FILE})
    message(VERBOSE "Generating ${kernel_name} kernels for the first time...")
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${generator_script} ${OUTPUT_BASE_DIR} ${kernel_name}
      RESULT_VARIABLE GENERATION_RESULT
      OUTPUT_VARIABLE GENERATION_OUTPUT
      ERROR_VARIABLE GENERATION_ERROR
    )

    if(NOT GENERATION_RESULT EQUAL 0)
      message(
        FATAL_ERROR
          "Failed to generate kernel files during configuration\nOutput: ${GENERATION_OUTPUT}\nError: ${GENERATION_ERROR}"
      )
    endif()
  endif()

  # Include the generated CMake file The generated file handles setting variables to PARENT_SCOPE
  include(${GENERATED_CMAKE_FILE})
endfunction()
