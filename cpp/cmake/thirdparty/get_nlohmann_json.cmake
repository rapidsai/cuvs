#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_and_configure_nlohmann_json)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(nlohmann_json ${PKG_VERSION}
            GLOBAL_TARGETS      nlohmann_json::nlohmann_json
            CPM_ARGS
            GIT_REPOSITORY         https://github.com/${PKG_FORK}/json.git
            GIT_TAG                ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            )

    if(glog_ADDED)
        message(VERBOSE "cuVS: Using glog located in ${glog_SOURCE_DIR}")
    else()
        message(VERBOSE "cuVS: Using glog located in ${glog_DIR}")
    endif()

endfunction()

find_and_configure_nlohmann_json(VERSION  3.11.2
        FORK             nlohmann
        PINNED_TAG       v3.11.2
        EXCLUDE_FROM_ALL ON
        )
