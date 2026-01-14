/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sample_filter.cuh"

namespace cuvs::neighbors::ivf_flat::detail {

template <typename index_t>
__device__ bool sample_filter(index_t* const* const inds_ptrs,
                              const uint32_t query_ix,
                              const uint32_t cluster_ix,
                              const uint32_t sample_ix,
                              uint32_t* bitset_ptr,
                              index_t bitset_len,
                              index_t original_nbits)
{
  auto bitset_view =
    raft::core::bitset_view<uint32_t, index_t>{bitset_ptr, bitset_len, original_nbits};
  auto bitset_filter = cuvs::neighbors::filtering::bitset_filter<uint32_t, index_t>{bitset_view};
  auto ivf_to_sample_filter = cuvs::neighbors::filtering::
    ivf_to_sample_filter<index_t, cuvs::neighbors::filtering::bitset_filter<uint32_t, index_t>>{
      inds_ptrs, bitset_filter};
  return ivf_to_sample_filter(query_ix, cluster_ix, sample_ix);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
