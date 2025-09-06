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

set(file_contents)
foreach(obj ${OBJECTS})
  get_filename_component(obj_ext ${obj} EXT)
  get_filename_component(obj_name ${obj} NAME_WE)
  get_filename_component(obj_dir ${obj} DIRECTORY)

  if(obj_ext MATCHES ".fatbin")
    set(args -c -p 0x0 --name embedded_${obj_name} ${obj})
    execute_process(COMMAND "${BIN_TO_C_COMMAND}" ${args}
                    WORKING_DIRECTORY ${obj_dir}
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE output
                    ERROR_VARIABLE error_var
                    )
    string(APPEND file_contents "\n${output}")
  endif()
endforeach()
file(WRITE "${OUTPUT}" "${file_contents}")
