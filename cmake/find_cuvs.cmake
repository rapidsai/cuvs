# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This file is copied to a temporary CMakeLists.txt by build.rs. It intentionally performs only the
# versioned cuVS package discovery step; Rust uses the selected cuvs_DIR with cmake-package for
# target introspection.

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)
project(cuvs_package_probe LANGUAGES C CXX)

if(NOT DEFINED OUTPUT_FILE)
  message(FATAL_ERROR "OUTPUT_FILE is not set")
endif()

if(NOT DEFINED REQUIRED_VERSION)
  message(FATAL_ERROR "REQUIRED_VERSION is not set")
endif()

if(NOT DEFINED CUVS_COMPONENT)
  set(CUVS_COMPONENT c_api)
endif()

function(json_set_string json_var key value)
  string(JSON _json SET "${${json_var}}" "${key}" "\"${value}\"")
  set(${json_var}
      "${_json}"
      PARENT_SCOPE
  )
endfunction()

set(_find_args cuvs ${REQUIRED_VERSION} CONFIG QUIET COMPONENTS ${CUVS_COMPONENT})
if(DEFINED CUVS_CMAKE_DIR)
  list(APPEND _find_args PATHS "${CUVS_CMAKE_DIR}" NO_DEFAULT_PATH)
endif()
find_package(${_find_args})

set(_considered "[]")
set(_index 0)
foreach(_config _version IN ZIP_LISTS cuvs_CONSIDERED_CONFIGS cuvs_CONSIDERED_VERSIONS)
  set(_candidate "{}")
  json_set_string(_candidate "config" "${_config}")
  json_set_string(_candidate "version" "${_version}")
  string(JSON _considered SET "${_considered}" ${_index} "${_candidate}")
  math(EXPR _index "${_index} + 1")
endforeach()

set(_json "{}")
string(JSON _json SET "${_json}" "considered" "${_considered}")

if(cuvs_FOUND)
  json_set_string(_json "cmake_dir" "${cuvs_DIR}")
endif()

file(WRITE "${OUTPUT_FILE}" "${_json}\n")
