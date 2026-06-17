# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

include("${CMAKE_CURRENT_LIST_DIR}/generate_cutile_cubins.cmake")

generate_cutile_vector_add_cubins(CUTILE_GENERATED_INCLUDE_DIR)

ConfigureTest(
  NAME CUTILE_VECTOR_ADD_TEST
  PATH "${CMAKE_CURRENT_LIST_DIR}/cutile_vector_add.cu"
  GPUS 1
  PERCENT 100
)

add_dependencies(CUTILE_VECTOR_ADD_TEST cutile_vector_add_cubins)

target_include_directories(
  CUTILE_VECTOR_ADD_TEST PRIVATE "${CUTILE_GENERATED_INCLUDE_DIR}"
)
