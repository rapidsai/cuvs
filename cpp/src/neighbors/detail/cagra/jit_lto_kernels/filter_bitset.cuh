/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../sample_filter.cuh"
#include "filter_data.h"
#include <raft/core/bitset.cuh>

namespace cuvs::neighbors::cagra::detail {

template <typename SourceIndexT>
__device__ bool sample_filter(uint32_t query_id, SourceIndexT node_id, void* filter_data)
{
  // bitset_filter checks if the node_id is in the bitset
  // filter_data points to bitset_filter_data_t<SourceIndexT> struct
  if (filter_data == nullptr) {
    return true;  // No filter data, allow all
  }

  auto* bitset_data = static_cast<bitset_filter_data_t<SourceIndexT>*>(filter_data);
  if (bitset_data->bitset_ptr == nullptr) {
    return true;  // No bitset provided, allow all
  }

  // Create bitset_view and filter, matching non-JIT behavior
  auto bitset_view = raft::core::bitset_view<uint32_t, SourceIndexT>{
    bitset_data->bitset_ptr, bitset_data->bitset_len, bitset_data->original_nbits};
  auto bitset_filter =
    cuvs::neighbors::filtering::bitset_filter<uint32_t, SourceIndexT>{bitset_view};
  return bitset_filter(query_id, node_id);
}

}  // namespace cuvs::neighbors::cagra::detail
