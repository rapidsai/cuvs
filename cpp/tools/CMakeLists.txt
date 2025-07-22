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
