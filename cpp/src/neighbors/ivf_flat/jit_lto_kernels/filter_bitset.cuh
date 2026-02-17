/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_flat::detail {

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

template <typename index_t>
__device__ bool sample_filter(index_t* const* const inds_ptrs,
                              const uint32_t query_ix,
                              const uint32_t cluster_ix,
                              const uint32_t sample_ix,
                              uint32_t* bitset_ptr,
                              index_t bitset_len,
                              index_t original_nbits)
{
  // Convert cluster_ix and sample_ix to a single sample index using inds_ptrs
  const index_t sample_index = inds_ptrs[cluster_ix][sample_ix];

  // Directly test the bitset without needing bitset_filter or ivf_to_sample_filter wrappers
  return bitset_view_test<uint32_t, index_t>(bitset_ptr, bitset_len, original_nbits, sample_index);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
