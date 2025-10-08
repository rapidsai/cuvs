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

# Read objects from response file to avoid argument length issues
if(DEFINED OBJECTS_RESPONSE_FILE)
  file(READ "${OBJECTS_RESPONSE_FILE}" objects_content)
  string(STRIP "${objects_content}" objects_content)
  # Split by newlines since we joined with \n in the CMake file
  string(REPLACE "\n" ";" objects_list "${objects_content}")
else()
  # Fallback to direct objects (for backward compatibility)
  set(objects_list "${OBJECTS}")
endif()

# Create output directory if it doesn't exist
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

set(generated_headers)
foreach(obj ${objects_list})
  # Skip empty entries
  if(NOT obj STREQUAL "")
    get_filename_component(obj_ext ${obj} EXT)
    get_filename_component(obj_name ${obj} NAME_WE)
    get_filename_component(obj_dir ${obj} DIRECTORY)

    if(obj_ext MATCHES ".fatbin")
      # Generate individual header file for this FATBIN
      set(header_file "${OUTPUT_DIR}/${obj_name}.h")

      set(args -c -p 0x0 --name embedded_${obj_name} ${obj})
      execute_process(
        COMMAND "${BIN_TO_C_COMMAND}" ${args}
        WORKING_DIRECTORY ${obj_dir}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
        ERROR_VARIABLE error_var
      )
      if(NOT result EQUAL 0)
        message(FATAL_ERROR "Failed to process ${obj}: ${error_var}")
      endif()

      # Write individual header file
      file(WRITE "${header_file}" "${output}")
      list(APPEND generated_headers "${header_file}")
    endif()
  endif()
endforeach()

# Create a stamp file to indicate completion
file(WRITE "${STAMP_FILE}" "Headers generated: ${generated_headers}")
list(LENGTH generated_headers num_headers)
message(VERBOSE "Generated ${num_headers} individual FATBIN headers")
