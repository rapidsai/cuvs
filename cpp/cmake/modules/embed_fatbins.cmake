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


function(embed_fatbins library_name kernel_target)
  find_package(CUDAToolkit REQUIRED)
  find_program(bin_to_c
    NAMES bin2c
    PATHS ${CUDAToolkit_BIN_DIR}
    )

  set(output_header ${CMAKE_CURRENT_BINARY_DIR}/${library_name}/embedded_fatbins.h)

  # Generates the header(s) with the inline fatbins
  add_custom_command(
    OUTPUT "${output_header}"
    COMMAND ${CMAKE_COMMAND}
      "-DBIN_TO_C_COMMAND=${bin_to_c}"
      "-DOBJECTS=$<TARGET_OBJECTS:${kernel_target}>"
      "-DOUTPUT=${output_header}"
      -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_header.cmake
    VERBATIM
    DEPENDS $<TARGET_OBJECTS:${kernel_target}>
    COMMENT "Converting FATBIN kernels to C++ header"
    )

  # get the sources of `kernel_target` and add them as CUDA
  # sources so we re-compile them to get the inline registration logic
  get_target_property(output_sources ${kernel_target} SOURCES)

  # add those c++ sources to `library_name`
  target_sources(${library_name}
    PRIVATE
    ${output_header}
    ${output_sources}
  )
  target_compile_features(${library_name} PRIVATE cxx_std_20)
  target_include_directories(${library_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${library_name})
endfunction()
