/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::detail {

// Structure to hold bitset filter data
// This is passed as void* to the extern sample_filter function
// Used by both CAGRA and IVF Flat
template <typename BitsetT, typename IndexT>
struct bitset_filter_data_t {
  BitsetT* bitset_ptr;    // Pointer to bitset data in global memory
  IndexT bitset_len;      // Length of bitset array
  IndexT original_nbits;  // Original number of bits

  __device__ bitset_filter_data_t(BitsetT* ptr, IndexT len, IndexT nbits)
    : bitset_ptr(ptr), bitset_len(len), original_nbits(nbits)
  {
  }
};

}  // namespace cuvs::neighbors::detail
