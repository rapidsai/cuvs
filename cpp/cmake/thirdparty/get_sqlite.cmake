#=============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
#=============================================================================

function(find_and_configure_sqlite)
    set(oneValueArgs VERSION YEAR)
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # SQLite amalgamation is distributed as a single .c file with a header
    # We'll fetch it and create a static library
    rapids_cpm_find(sqlite3 ${PKG_VERSION}
            GLOBAL_TARGETS      sqlite3
            CPM_ARGS
            URL                https://www.sqlite.org/${PKG_YEAR}/sqlite-amalgamation-${PKG_VERSION}.zip
            DOWNLOAD_ONLY      YES
            )

    if(sqlite3_ADDED)
        message(VERBOSE "cuVS: Using SQLite3 amalgamation from ${sqlite3_SOURCE_DIR}")

        # Create a static library from the amalgamation
        add_library(sqlite3 STATIC ${sqlite3_SOURCE_DIR}/sqlite3.c)

        target_include_directories(sqlite3 PUBLIC
            $<BUILD_INTERFACE:${sqlite3_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        )

        set_target_properties(sqlite3 PROPERTIES EXCLUDE_FROM_ALL ON)
    else()
        message(VERBOSE "cuVS: Using SQLite3 located in ${sqlite3_DIR}")
    endif()

endfunction()

find_and_configure_sqlite(
    VERSION 3470200
    YEAR    2024
)
