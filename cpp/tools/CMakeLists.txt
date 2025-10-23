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

# Force CMake to use conda clang installation if available
if(DEFINED ENV{CONDA_PREFIX})
  set(Clang_DIR "$ENV{CONDA_PREFIX}/lib/cmake/clang")
  set(LLVM_DIR "$ENV{CONDA_PREFIX}/lib/cmake/llvm")
  message(STATUS "Using conda environment: $ENV{CONDA_PREFIX}")
  message(STATUS "Clang_DIR: ${Clang_DIR}")
  message(STATUS "LLVM_DIR: ${LLVM_DIR}")
endif()

# Use rapids_find_package to require Clang and LLVM
rapids_find_package(
  Clang REQUIRED
  BUILD_EXPORT_SET cuvs-tools-exports
  INSTALL_EXPORT_SET cuvs-tools-exports
)

rapids_find_package(
  LLVM REQUIRED
  BUILD_EXPORT_SET cuvs-tools-exports
  INSTALL_EXPORT_SET cuvs-tools-exports
)

# Add LLVM and Clang definitions and include directories
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

# Create the generic JSON generator tool
add_executable(generate_generic_json generate_generic_json.cpp)

# Link against required Clang libraries
target_link_libraries(
  generate_generic_json clangTooling clangBasic clangAST clangFrontend clangSerialization
  clangDriver clangParse clangSema clangLex
)

# Set C++ standard
set_target_properties(generate_generic_json PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Set up output directory for generated files
set(CUVS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(CUVS_GENERATED_TARGETS_FILE "${CUVS_GENERATED_DIR}/generated_targets.cmake")

# Create output directory
file(MAKE_DIRECTORY "${CUVS_GENERATED_DIR}")

# Create empty generated targets file early so CMake can include it and track changes
if(NOT EXISTS "${CUVS_GENERATED_TARGETS_FILE}")
  file(
    WRITE "${CUVS_GENERATED_TARGETS_FILE}"
    "# Generated CMake file for generic JSON serialization
# AUTO-GENERATED - DO NOT EDIT
#
# This empty file is created early in configuration to allow CMake
# to properly track changes and trigger rebuilds when content is generated.
#
"
  )
endif()

# Create a custom command to run the code generation
add_custom_command(
  OUTPUT "${CUVS_GENERATED_TARGETS_FILE}"
  COMMAND generate_generic_json --output-dir "${CUVS_GENERATED_DIR}" -p
          "${CMAKE_CURRENT_BINARY_DIR}/.." --manifest "${CUVS_GENERATED_TARGETS_FILE}"
  DEPENDS generate_generic_json "${CMAKE_CURRENT_BINARY_DIR}/../compile_commands.json"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Generating generic JSON serialization code"
  VERBATIM
)

# Create a custom target that depends on the manifest generation
add_custom_target(cuvs_generate_generic_code ALL DEPENDS "${CUVS_GENERATED_TARGETS_FILE}")

# Function to include the generated CMake file
function(include_generated_targets)
  if(EXISTS "${CUVS_GENERATED_TARGETS_FILE}")
    message(STATUS "Including generated CMake targets from: ${CUVS_GENERATED_TARGETS_FILE}")
    include("${CUVS_GENERATED_TARGETS_FILE}")
  else()
    message(STATUS "Generated CMake file not found - will be created during build")
  endif()
endfunction()

# Try to include existing generated targets from previous build, or they will be created during
# build
include_generated_targets()
