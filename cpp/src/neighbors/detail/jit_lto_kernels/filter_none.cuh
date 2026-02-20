/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::detail {

// Unified sample_filter: takes query_id, node_id, and void* filter_data
// Used by both CAGRA and IVF Flat
template <typename SourceIndexT>
__device__ bool sample_filter(uint32_t query_id, SourceIndexT node_id, void* filter_data)
{
  // none_sample_filter always returns true (no filtering)
  // filter_data is ignored (can be nullptr)
  return true;
}

}  // namespace cuvs::neighbors::detail
