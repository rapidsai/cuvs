# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# Set the list of Cython files to build
set(cython_sources distance.pyx)
set(linked_libraries cuvs::cuvs cuvs::c_api)

# Build all of the Cython targets
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX distance_
)
