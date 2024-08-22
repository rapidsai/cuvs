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
set(RAFT_VERSION "${RAPIDS_VERSION_MAJOR_MINOR}")
set(RAFT_FORK "rapidsai")
set(RAFT_PINNED_TAG "branch-${RAPIDS_VERSION_MAJOR_MINOR}")

function(find_and_configure_raft)
    set(oneValueArgs VERSION FORK PINNED_TAG USE_RAFT_STATIC ENABLE_NVTX ENABLE_MNMG_DEPENDENCIES CLONE_ON_PIN)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "branch-${RAFT_VERSION}")
        message(STATUS "cuVS: RAFT pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
        set(CPM_DOWNLOAD_raft ON)
    elseif(PKG_USE_RAFT_STATIC AND (NOT CPM_raft_SOURCE))
        message(STATUS "cuVS: Cloning raft locally to build static libraries.")
        set(CPM_DOWNLOAD_raft ON)
    endif()

    set(RAFT_COMPONENTS "")

    if(PKG_ENABLE_MNMG_DEPENDENCIES)
        string(APPEND RAFT_COMPONENTS " distributed")
    endif()

    #-----------------------------------------------------
    # Invoke CPM find_package()
    #-----------------------------------------------------
    rapids_cpm_find(raft ${PKG_VERSION}
            GLOBAL_TARGETS      raft::raft
            BUILD_EXPORT_SET    cuvs-exports
            INSTALL_EXPORT_SET  cuvs-exports
            COMPONENTS          ${RAFT_COMPONENTS}
            CPM_ARGS
              GIT_REPOSITORY        https://github.com/${PKG_FORK}/raft.git
              GIT_TAG               ${PKG_PINNED_TAG}
              SOURCE_SUBDIR         cpp
              OPTIONS
              "BUILD_TESTS OFF"
              "BUILD_PRIMS_BENCH OFF"
              "BUILD_ANN_BENCH OFF"
              "RAFT_NVTX ${PKG_ENABLE_NVTX}"
              "RAFT_COMPILE_LIBRARY OFF"
            )
endfunction()


# Change pinned tag here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION  ${RAFT_VERSION}.00
        FORK                     ${RAFT_FORK}
        PINNED_TAG               ${RAFT_PINNED_TAG}
        ENABLE_MNMG_DEPENDENCIES OFF
        ENABLE_NVTX              OFF
        USE_RAFT_STATIC ${CUVS_USE_RAFT_STATIC}
        # When PINNED_TAG above doesn't match the default rapids branch,
        # force local raft clone in build directory
        # even if it's already installed.
        CLONE_ON_PIN     ${CUVS_RAFT_CLONE_ON_PIN}

)
