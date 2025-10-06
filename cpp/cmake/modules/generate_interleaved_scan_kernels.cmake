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

# Generate interleaved scan kernel files at build time
function(generate_interleaved_scan_kernels)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  set(KERNEL_LIST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_kernels.txt)
  set(GENERATOR_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/generate_kernels.py)
  set(OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/neighbors/ivf_flat/jit_lto_kernels/interleaved_scan_kernels)
  set(CMAKE_LIST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/jit_lto_kernels_list/interleaved_scan.cmake)
  set(STAMP_FILE ${CMAKE_CURRENT_BINARY_DIR}/kernels_generated.stamp)

  # Generate the kernels at build time
  add_custom_command(
    OUTPUT ${STAMP_FILE}
    COMMAND ${Python3_EXECUTABLE} ${GENERATOR_SCRIPT}
    COMMAND ${CMAKE_COMMAND} -E touch ${STAMP_FILE}
    DEPENDS ${KERNEL_LIST_FILE} ${GENERATOR_SCRIPT}
    COMMENT "Generating interleaved scan kernel files..."
    VERBATIM
  )

  # Create a custom target that depends on the stamp file
  add_custom_target(generate_interleaved_scan_kernels_target
    DEPENDS ${STAMP_FILE}
  )

  # Include the generated CMake list file
  # Only generate if the CMake list file doesn't exist
  if(NOT EXISTS ${CMAKE_LIST_FILE})
    message(STATUS "Generating interleaved scan kernels for the first time...")
    execute_process(
      COMMAND ${Python3_EXECUTABLE} ${GENERATOR_SCRIPT}
      RESULT_VARIABLE GENERATION_RESULT
      OUTPUT_VARIABLE GENERATION_OUTPUT
      ERROR_VARIABLE GENERATION_ERROR
    )

    if(NOT GENERATION_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to generate kernel files during configuration\nOutput: ${GENERATION_OUTPUT}\nError: ${GENERATION_ERROR}")
    endif()
  endif()

  # Include the generated CMake file
  include(${CMAKE_LIST_FILE})

  # Prepend the source directory path to all kernel files
  set(FULL_PATH_KERNEL_FILES)
  foreach(kernel_file ${INTERLEAVED_SCAN_KERNEL_FILES})
    list(APPEND FULL_PATH_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${kernel_file})
  endforeach()

  # Prepend the source directory path to all metric device function files
  set(FULL_PATH_METRIC_FILES)
  foreach(metric_file ${METRIC_DEVICE_FUNCTION_FILES})
    list(APPEND FULL_PATH_METRIC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${metric_file})
  endforeach()

  # Return the lists to parent scope
  set(INTERLEAVED_SCAN_KERNEL_FILES ${FULL_PATH_KERNEL_FILES} PARENT_SCOPE)
  set(METRIC_DEVICE_FUNCTION_FILES ${FULL_PATH_METRIC_FILES} PARENT_SCOPE)
  set(INTERLEAVED_SCAN_KERNELS_STAMP ${STAMP_FILE} PARENT_SCOPE)
  set(INTERLEAVED_SCAN_KERNELS_TARGET generate_interleaved_scan_kernels_target PARENT_SCOPE)
endfunction()
