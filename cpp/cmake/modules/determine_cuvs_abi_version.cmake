# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================
include_guard(GLOBAL)

#[=======================================================================[.rst:
determine_cuvs_abi_version
---------------------------

.. versionadded:: v26.02.00

Convert a calendar version to a cuVS ABI version

  .. code-block:: cmake

    determine_cuvs_abi_version(cal_ver MAJOR major_output_var MINOR minor_output_var)

Provides a consistent method to convert calendar-based version strings (YY.MM format) to
cuVS ABI version components.

Each

``cal_ver``
    A calendar version string in YY.MM format (e.g., "26.02", "27.08").

``major_output_var``
    Contains the name of the variable that will be set in the parent scope to the computed
    ABI major version.

``minor_output_var``
    Contains the name of the variable that will be set in the parent scope to the computed
    ABI minor version.

Example on how to properly use :cmake:command:`determine_cuvs_abi_version`:

  .. code-block:: cmake

    project(Example VERSION 26.02.0)

    determine_cuvs_abi_version(${PROJECT_VERSION} MAJOR abi_major MINOR abi_minor)
    message(STATUS "CalVer ${calver} maps to ABI ${abi_major}.${abi_minor}")


Result Variables
^^^^^^^^^^^^^^^^
  Variables matching the contents of ``major_output_var`` and ``minor_output_var`` will be set in the parent
  scope with the computed ABI version components.

  ``${major_output_var}``
      Contains the ABI major version component

  ``${minor_output_var}``
      Contains the ABI minor version component

#]=======================================================================]
# cmake-lint: disable=C0112
function(determine_cuvs_abi_version cal_ver)
  set(options)
  set(one_value "MAJOR" "MINOR")
  set(multi_value)
  cmake_parse_arguments(_CUVS_RAPIDS "${options}" "${one_value}" "${multi_value}" ${ARGN})

  rapids_cmake_parse_version(MAJOR ${cal_ver} cal_ver_major)
  rapids_cmake_parse_version(MINOR ${cal_ver} cal_ver_minor)

  # Encode the last ABI break
  set(current_major_abi_ver "1") # The current ABI major value
  set(abi_base_year "26") # What year the current ABI major occurred in
  set(abi_base_month "04") # What month the current ABI major occurred in
  # compute the abi version
  if(cal_ver_major STREQUAL abi_base_year)
    # If we are in the same year is is pretty easy to compute our abi break
    math(EXPR computed_abi_minor "(${cal_ver_minor}-${abi_base_month})/2")
  else()
    #
    math(EXPR first_year_count "(12-${abi_base_month})/2")
    math(EXPR extra_years "(${cal_ver_major} - ${abi_base_year} - 1) * 6")
    math(EXPR this_year_count "(${cal_ver_minor})/2")
    math(EXPR computed_abi_minor "${first_year_count} + ${extra_years} + ${this_year_count}")
  endif()

  set(${_CUVS_RAPIDS_MAJOR}
      ${computed_abi_major}
      PARENT_SCOPE
  )
  set(${_CUVS_RAPIDS_MINOR}
      ${computed_abi_minor}
      PARENT_SCOPE
  )
endfunction()
