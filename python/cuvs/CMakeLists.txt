# =============================================================================
# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

# We always need CUDA for cuvs because the cuvs dependency brings in a header-only cuco dependency
# that enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(cuvs_py)

project(
  cuvs_py
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES # TODO: Building Python extension modules via the python_extension_module requires the C
            # language to be enabled here. The test project that is built in scikit-build to verify
            # various linking options for the python library is hardcoded to build with C, so until
            # that is fixed we need to keep C.
            C CXX CUDA
)

# ##################################################################################################
# * Process User Options  ------------------------------------------------------

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cython-core)
include(rapids-export)
include(rapids-find)

rapids_cpm_init()

# --- cuVS ---#
find_package(cuvs "${RAPIDS_VERSION}" REQUIRED COMPONENTS c_api)

# --- dlpack ---#
include(../../cpp/cmake/thirdparty/get_dlpack.cmake)

# ensure Cython targets can find dlpack headers (these do not come installed with with cuVS)
target_include_directories(cuvs::cuvs INTERFACE "$<BUILD_INTERFACE:${DLPACK_INCLUDE_DIR}>")

# ##################################################################################################
# * Build Cython artifacts -----------------------------------------------------

rapids_cython_init()

add_subdirectory(cuvs/common)
add_subdirectory(cuvs/cluster)
add_subdirectory(cuvs/distance)
add_subdirectory(cuvs/neighbors)
add_subdirectory(cuvs/preprocessing)
