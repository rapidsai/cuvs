/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/haversine_distance.cuh"  // compute_haversine
#include <cuvs/distance/distance.hpp>

#include <cstdint>  // uint32_t

namespace cuvs::neighbors::ball_cover::detail {

template <typename value_t, typename ValueInt>  // NOLINT(readability-identifier-naming)
struct dist_func {
  virtual __device__ __host__ __forceinline__ auto operator()(const value_t* a,
                                                              const value_t* b,
                                                              const ValueInt n_dims) -> value_t
  {
    return -1;
  };
};

template <typename value_t, typename ValueInt>  // NOLINT(readability-identifier-naming)
struct haversine_func : public dist_func<value_t, ValueInt> {
  __device__ __host__ __forceinline__ auto operator()(const value_t* a,
                                                      const value_t* b,
                                                      const ValueInt n_dims) -> value_t override
  {
    return cuvs::neighbors::detail::compute_haversine<value_t, value_t>(a[0], b[0], a[1], b[1]);
  }
};

template <typename value_t, typename ValueInt>  // NOLINT(readability-identifier-naming)
struct euclidean_func : public dist_func<value_t, ValueInt> {
  __device__ __host__ __forceinline__ auto operator()(const value_t* a,
                                                      const value_t* b,
                                                      const ValueInt n_dims) -> value_t override
  {
    value_t sum_sq = 0;
    for (ValueInt i = 0; i < n_dims; ++i) {
      value_t diff = a[i] - b[i];
      sum_sq += diff * diff;
    }

    return raft::sqrt(sum_sq);
  }
};

template <typename value_t, typename ValueInt>  // NOLINT(readability-identifier-naming)
struct euclidean_sq_func : public dist_func<value_t, ValueInt> {
  __device__ __host__ __forceinline__ auto operator()(const value_t* a,
                                                      const value_t* b,
                                                      const ValueInt n_dims) -> value_t override
  {
    value_t sum_sq = 0;
    for (ValueInt i = 0; i < n_dims; ++i) {
      value_t diff = a[i] - b[i];
      sum_sq += diff * diff;
    }
    return sum_sq;
  }
};

// Direct distance function for use in kernels that need metric information
template <typename value_t, typename ValueInt>  // NOLINT(readability-identifier-naming)
__device__ __host__ __forceinline__ auto compute_distance_by_metric(
  const value_t* a, const value_t* b, const ValueInt n_dims, cuvs::distance::DistanceType metric)
  -> value_t
{
  if (metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      metric == cuvs::distance::DistanceType::L2SqrtUnexpanded) {
    // Euclidean distance implementation
    return euclidean_func<value_t, ValueInt>{}(a, b, n_dims);
  } else if (metric == cuvs::distance::DistanceType::Haversine) {
    // Haversine distance implementation
    return haversine_func<value_t, ValueInt>{}(a, b, n_dims);
  }

  // Default case
  return -1;
}

};  // namespace cuvs::neighbors::ball_cover::detail
