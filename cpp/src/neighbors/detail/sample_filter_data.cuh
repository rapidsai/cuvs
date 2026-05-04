/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::detail {

// Passed as void* to the unified JIT sample_filter (see sample_filter.cuh / bitset_filter).
template <typename SourceIndexT>
struct bitset_filter_data_t {
  uint32_t* bitset_ptr;
  SourceIndexT bitset_len;
  SourceIndexT original_nbits;

  __device__ bitset_filter_data_t(uint32_t* ptr, SourceIndexT len, SourceIndexT nbits)
    : bitset_ptr(ptr), bitset_len(len), original_nbits(nbits)
  {
  }
};

}  // namespace cuvs::neighbors::detail
