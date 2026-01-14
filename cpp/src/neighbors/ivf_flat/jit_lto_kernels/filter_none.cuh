/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sample_filter.cuh"

namespace cuvs::neighbors::ivf_flat::detail {

template <typename index_t>
__device__ constexpr bool sample_filter(index_t* const* const inds_ptrs,
                                        const uint32_t query_ix,
                                        const uint32_t cluster_ix,
                                        const uint32_t sample_ix,
                                        uint32_t* bitset_ptr,
                                        index_t bitset_len,
                                        index_t original_nbits)
{
  return true;
}

}  // namespace cuvs::neighbors::ivf_flat::detail
