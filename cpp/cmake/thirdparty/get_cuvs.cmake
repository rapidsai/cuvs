# =============================================================================
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

# Use RAPIDS_VERSION_MAJOR_MINOR from rapids_config.cmake
set(CUVS_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(CUVS_FORK "rapidsai")
set(CUVS_PINNED_TAG "branch-${RAPIDS_VERSION_MAJOR_MINOR}")

function(find_and_configure_cuvs)
    set(oneValueArgs VERSION FORK PINNED_TAG ENABLE_NVTX CLONE_ON_PIN BUILD_CPU_ONLY BUILD_SHARED_LIBS)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${CUVS_VERSION}")
        message(STATUS "cuVS: pinned tag found: ${PKG_PINNED_TAG}. Cloning cuVS locally.")
        set(CPM_DOWNLOAD_cuvs ON)
    endif()

    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(cuvs ${PKG_VERSION}
            GLOBAL_TARGETS      cuvs::cuvs
            BUILD_EXPORT_SET    cuvs-bench-exports
            INSTALL_EXPORT_SET  cuvs-bench-exports
            COMPONENTS          cuvs
            CPM_ARGS
              GIT_REPOSITORY        https://github.com/${PKG_FORK}/cuvs.git
              GIT_TAG               ${PKG_PINNED_TAG}
              SOURCE_SUBDIR         cpp
              OPTIONS
              "BUILD_SHARED_LIBS ${PKG_BUILD_SHARED_LIBS}"
              "BUILD_CPU_ONLY ${PKG_BUILD_CPU_ONLY}"
              "BUILD_TESTS OFF"
              "BUILD_CAGRA_HNSWLIB OFF"
              "CUVS_CLONE_ON_PIN ${PKG_CLONE_ON_PIN}"
            )
endfunction()


# Change pinned tag here to test a commit in CI
# To use a different cuVS locally, set the CMake variable
# CPM_cuvs_SOURCE=/path/to/local/cuvs
find_and_configure_cuvs(VERSION  ${CUVS_VERSION}.00
        FORK                     ${CUVS_FORK}
        PINNED_TAG               ${CUVS_PINNED_TAG}
        ENABLE_NVTX              OFF
        # When PINNED_TAG above doesn't match the default rapids branch,
        # force local cuvs clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ${CUVS_CLONE_ON_PIN}
        BUILD_CPU_ONLY ${BUILD_CPU_ONLY}
        BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}
)
