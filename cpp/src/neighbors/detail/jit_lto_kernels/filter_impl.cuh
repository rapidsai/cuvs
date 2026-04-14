/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "filter.cuh"
#include "filter_data.cuh"

namespace cuvs::neighbors::detail {

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_none(uint32_t query_id, IndexT node_id, void* filter_data)
{
  return true;
}

// Inline implementation of bitset_view::test() to avoid including bitset.cuh
// which transitively includes Thrust
template <typename BitsetT, typename IndexT>
__device__ inline bool bitset_view_test(const BitsetT* bitset_ptr,
                                        IndexT bitset_len,
                                        IndexT original_nbits,
                                        IndexT sample_index)
{
  constexpr IndexT bitset_element_size = sizeof(BitsetT) * 8;
  const IndexT nbits                   = sizeof(BitsetT) * 8;
  IndexT bit_index                     = 0;
  IndexT bit_offset                    = 0;

  if (original_nbits == 0 || nbits == original_nbits) {
    bit_index  = sample_index / bitset_element_size;
    bit_offset = sample_index % bitset_element_size;
  } else {
    // Handle original_nbits != nbits case
    const IndexT original_bit_index  = sample_index / original_nbits;
    const IndexT original_bit_offset = sample_index % original_nbits;
    bit_index                        = original_bit_index * original_nbits / nbits;
    bit_offset                       = 0;
    if (original_nbits > nbits) {
      bit_index += original_bit_offset / nbits;
      bit_offset = original_bit_offset % nbits;
    } else {
      IndexT ratio = nbits / original_nbits;
      bit_offset += (original_bit_index % ratio) * original_nbits;
      bit_offset += original_bit_offset % nbits;
    }
  }
  const BitsetT bit_element = bitset_ptr[bit_index];
  const bool is_bit_set     = (bit_element & (BitsetT{1} << bit_offset)) != 0;
  return is_bit_set;
}

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_bitset(uint32_t query_id, IndexT node_id, void* filter_data)
{
  // bitset_filter checks if the node_id is in the bitset
  // filter_data points to bitset_filter_data_t<SourceIndexT> struct
  if (filter_data == nullptr) {
    return true;  // No filter data, allow all
  }

  auto* bitset_data = static_cast<bitset_filter_data_t<BitsetT, IndexT>*>(filter_data);
  if (bitset_data->bitset_ptr == nullptr) {
    return true;  // No bitset provided, allow all
  }

  // Directly test the bitset without needing bitset_filter wrapper
  // bitset_view_test returns true if the bit is set (node_id is in the bitset)
  // The bitset marks allowed indices (same as non-JIT bitset_filter which returns test() directly)
  // Return true if the bit is set (node is allowed), false if not set (node should be filtered out)
  bool is_in_bitset = bitset_view_test<BitsetT, IndexT>(
    bitset_data->bitset_ptr, bitset_data->bitset_len, bitset_data->original_nbits, node_id);
  // If node_id is in the bitset (allowed), return true to allow it
  // If node_id is not in the bitset, return false to reject it
  return is_in_bitset;
}

}  // namespace cuvs::neighbors::detail
