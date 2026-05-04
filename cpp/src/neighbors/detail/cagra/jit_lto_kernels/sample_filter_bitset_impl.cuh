/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sample_filter_data.cuh"

#include <raft/core/bitset.cuh>

#include <cstdint>

namespace cuvs::neighbors::detail {

// JIT LTO: unified 3-arg device hook. Semantics match filtering::bitset_filter::operator()
// (query_ix, sample_ix) in sample_filter.cuh — return bitset_view.test(sample_ix); here
// sample_ix is the dataset node id.
template <typename SourceIndexT>
__device__ bool sample_filter(uint32_t /*query_id*/, SourceIndexT node_id, void* filter_data)
{
  if (filter_data == nullptr) { return true; }

  auto* data = static_cast<bitset_filter_data_t<SourceIndexT>*>(filter_data);
  if (data->bitset_ptr == nullptr) { return true; }

  raft::core::bitset_view<uint32_t, SourceIndexT> const view{
    data->bitset_ptr, data->bitset_len, data->original_nbits};
  return view.test(node_id);
}

}  // namespace cuvs::neighbors::detail
