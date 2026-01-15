# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

#[=======================================================================[.rst:
determine_cuvs_abi_version
---------------------------

.. versionadded:: v26.02.00

Convert a calendar version to a cuVS ABI version

  .. code-block:: cmake

    determine_cuvs_abi_version(cal_ver MAJOR major_output_var MINOR minor_output_var)

Provides a consistent method to convert calendar-based version strings (YY.MM format) to
cuVS ABI version components.

The conversion follows this mapping scheme:

- YY.02 releases map to ABI ?.0
- YY.04 releases map to ABI ?.1
- YY.06 releases map to ABI ?.2
- YY.08 releases map to ABI ?+1.0
- YY.10 releases map to ABI ?+1.1
- YY.12 releases map to ABI ?+1.2

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

    # Example output for various calendar versions:
    # 26.02 -> ABI 1.0
    # 26.08 -> ABI 2.0
    # 27.04 -> ABI 3.1


Result Variables
^^^^^^^^^^^^^^^^
  Variables matching the contents of ``major_output_var`` and ``minor_output_var`` will be set in the parent
  scope with the computed ABI version components.

  ``${major_output_var}``
      Contains the ABI major version component

  ``${minor_output_var}``
      Contains the ABI minor version component


#]=======================================================================]
function(determine_cuvs_abi_version cal_ver MAJOR _CUVS_RAPIDS_MAJOR_VAR MINOR _CUVS_RAPIDS_MINOR_VAR)
  # major value table
  set(major_table_02 "1")
  set(major_table_04 "1")
  set(major_table_06 "1")
  set(major_table_08 "2")
  set(major_table_10 "2")
  set(major_table_12 "2")
  # minor value table
  set(minor_table_02 "0")
  set(minor_table_04 "1")
  set(minor_table_06 "2")
  set(minor_table_08 "0")
  set(minor_table_10 "1")
  set(minor_table_12 "2")

  rapids_cmake_parse_version(MAJOR ${cal_ver} cal_ver_major)
  rapids_cmake_parse_version(MINOR ${cal_ver} cal_ver_minor)

  # compute the abi version
  math(EXPR computed_abi_major "(${cal_ver_major} - 26) * 2 + ${major_table_${cal_ver_minor}}")
  set(compute_abi_minor "${minor_table_${cal_ver_minor}}")


  set("${_CUVS_RAPIDS_MAJOR_VAR}" ${computed_abi_major} PARENT_SCOPE)
  set("${_CUVS_RAPIDS_MINOR_VAR}" ${compute_abi_minor} PARENT_SCOPE)
endfunction()
