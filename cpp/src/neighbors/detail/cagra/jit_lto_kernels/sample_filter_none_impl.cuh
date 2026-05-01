/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::detail {

// JIT LTO: unified 3-arg device hook. Semantics match filtering::none_sample_filter in
// sample_filter.cuh (always allow).
template <typename SourceIndexT>
__device__ bool sample_filter(uint32_t /*query_id*/,
                              SourceIndexT /*node_id*/,
                              void* /*filter_data*/)
{
  return true;
}

}  // namespace cuvs::neighbors::detail
