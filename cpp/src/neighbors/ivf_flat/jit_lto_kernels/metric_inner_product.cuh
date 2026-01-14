/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_utils.cuh>

namespace cuvs::neighbors::ivf_flat::detail {

template <int Veclen, typename T, typename AccT>
struct inner_prod_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    if constexpr (Veclen > 1 && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
      acc = raft::dp4a(x, y, acc);
    } else {
      acc += x * y;
    }
  }
};

template <int Veclen, typename T, typename AccT>
__device__ void compute_dist(AccT& acc, AccT x, AccT y)
{
  inner_prod_dist<Veclen, T, AccT>{}(acc, x, y);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
