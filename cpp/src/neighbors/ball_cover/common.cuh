/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/functional.h>
#include <thrust/tuple.h>

#include <cstdint>

namespace cuvs::neighbors::ball_cover::detail {

struct nn_comp {
  template <typename One, typename Two>
  __host__ __device__ auto operator()(const One& t1, const Two& t2) -> bool
  {
    // sort first by each sample's reference landmark,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then by closest neighbor,
    return thrust::get<1>(t1) < thrust::get<1>(t2);
  }
};

/**
 * Zeros the bit at location h in a One-hot encoded 32-bit int array
 */
__device__ inline void zero_bit(std::uint32_t* arr, std::uint32_t h)
{
  int bit = h % 32;
  int idx = h / 32;

  std::uint32_t assumed;
  std::uint32_t old = arr[idx];
  do {
    assumed = old;
    old     = atomicCAS(arr + idx, assumed, assumed & ~(1 << bit));
  } while (assumed != old);
}

/**
 * Returns whether or not bit at location h is nonzero in a One-hot
 * encoded 32-bit in array.
 */
__device__ inline auto get_val(std::uint32_t* arr, std::uint32_t h) -> bool
{
  int bit = h % 32;
  int idx = h / 32;
  return (arr[idx] & (1 << bit)) > 0;
}

};  // namespace cuvs::neighbors::ball_cover::detail
