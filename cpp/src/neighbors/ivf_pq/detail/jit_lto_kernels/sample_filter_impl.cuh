/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../sample_filter.cuh"

template <typename FilterT>
__device__ bool sample_filter_impl(const int64_t* const* const inds_ptrs,
                                   const uint32_t query_ix,
                                   const uint32_t cluster_ix,
                                   const uint32_t sample_ix,
                                   uint32_t* bitset_ptr,
                                   int64_t bitset_len,
                                   int64_t original_nbits)
{
  if constexpr (std::is_same_v<FilterT, cuvs::neighbors::filtering::none_sample_filter>) {
    return true;
  }

  if constexpr (std::is_same_v<FilterT,
                               cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>>) {
    auto bitset_view =
      raft::core::bitset_view<uint32_t, int64_t>{bitset_ptr, bitset_len, original_nbits};
    auto bitset_filter = cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>{bitset_view};
    auto ivf_to_sample_filter = cuvs::neighbors::filtering::
      ivf_to_sample_filter<int64_t, cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>>{
        inds_ptrs, bitset_filter};
    return ivf_to_sample_filter(query_ix, cluster_ix, sample_ix);
  }
}
