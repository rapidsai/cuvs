/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/matrix/detail/select_warpsort.cuh>  // matrix::detail::select::warpsort::warp_sort_distributed

namespace cuvs::neighbors::ivf::detail {

/**
 * Dummy block sort type used when Capacity is 0 in JIT kernels.
 * This is a minimal header that doesn't include CUB to avoid EmptyKernel instantiation.
 */
template <typename T, typename IdxT, bool Ascending = true>
struct dummy_block_sort_t {
  using queue_t = raft::matrix::detail::select::warpsort::
    warp_sort_distributed<raft::WarpSize, Ascending, T, IdxT>;
  template <typename... Args>
  __device__ dummy_block_sort_t(int k, Args...) {};
};

}  // namespace cuvs::neighbors::ivf::detail
