/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::ivf_sq::detail {

// Forward declaration of the sample filter device function.
// The concrete implementation is provided by a JIT-LTO filter-adapter fragment
// (see filter_kernel.cu.in) that delegates to the shared
// cuvs::neighbors::detail::sample_filter_<name><uint32_t, int64_t> fragment.
template <typename IndexT>
__device__ bool sample_filter(const IndexT* const* const inds_ptrs,
                              const uint32_t query_ix,
                              const uint32_t cluster_ix,
                              const uint32_t sample_ix,
                              uint32_t* bitset_ptr,
                              IndexT bitset_len,
                              IndexT original_nbits);

}  // namespace cuvs::neighbors::ivf_sq::detail
