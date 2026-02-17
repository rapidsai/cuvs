/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "filter_data.h"

namespace cuvs::neighbors::cagra::detail {

// Inline implementation of bitset_view::test() to avoid including bitset.cuh
// which transitively includes Thrust
template <typename bitset_t, typename index_t>
__device__ inline bool bitset_view_test(const bitset_t* bitset_ptr,
                                        index_t bitset_len,
                                        index_t original_nbits,
                                        index_t sample_index)
{
  constexpr index_t bitset_element_size = sizeof(bitset_t) * 8;
  const index_t nbits                   = sizeof(bitset_t) * 8;
  index_t bit_index                     = 0;
  index_t bit_offset                    = 0;

  if (original_nbits == 0 || nbits == original_nbits) {
    bit_index  = sample_index / bitset_element_size;
    bit_offset = sample_index % bitset_element_size;
  } else {
    // Handle original_nbits != nbits case
    const index_t original_bit_index  = sample_index / original_nbits;
    const index_t original_bit_offset = sample_index % original_nbits;
    bit_index                         = original_bit_index * original_nbits / nbits;
    bit_offset                        = 0;
    if (original_nbits > nbits) {
      bit_index += original_bit_offset / nbits;
      bit_offset = original_bit_offset % nbits;
    } else {
      index_t ratio = nbits / original_nbits;
      bit_offset += (original_bit_index % ratio) * original_nbits;
      bit_offset += original_bit_offset % nbits;
    }
  }
  const bitset_t bit_element = bitset_ptr[bit_index];
  const bool is_bit_set      = (bit_element & (bitset_t{1} << bit_offset)) != 0;
  return is_bit_set;
}

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

  // Directly test the bitset without needing bitset_filter wrapper
  // bitset_view_test returns true if the bit is set (node_id is in the bitset)
  // For a bitset created from removed_indices, if the bit is set, the node should be filtered out
  // So we return the inverse: if the bit is set, return false to reject the node
  bool is_in_bitset = bitset_view_test<uint32_t, SourceIndexT>(
    bitset_data->bitset_ptr, bitset_data->bitset_len, bitset_data->original_nbits, node_id);
  // If node_id is in the bitset (removed set), return false to reject it
  // If node_id is not in the bitset, return true to allow it
  return !is_in_bitset;
}

}  // namespace cuvs::neighbors::cagra::detail
