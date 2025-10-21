#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_and_configure_glog)
    set(oneValueArgs VERSION FORK PINNED_TAG EXCLUDE_FROM_ALL)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN} )

    rapids_cpm_find(glog ${PKG_VERSION}
            GLOBAL_TARGETS      glog::glog
            CPM_ARGS
            GIT_REPOSITORY         https://github.com/${PKG_FORK}/glog.git
            GIT_TAG                ${PKG_PINNED_TAG}
            EXCLUDE_FROM_ALL       ${PKG_EXCLUDE_FROM_ALL}
            )

    if(glog_ADDED)
        message(VERBOSE "cuVS: Using glog located in ${glog_SOURCE_DIR}")
    else()
        message(VERBOSE "cuVS: Using glog located in ${glog_DIR}")
    endif()

endfunction()

find_and_configure_glog(VERSION 0.6.0
        FORK             google
        PINNED_TAG       v0.6.0
        EXCLUDE_FROM_ALL ON
        )
