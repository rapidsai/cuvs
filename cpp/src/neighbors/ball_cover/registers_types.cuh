/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "../detail/haversine_distance.cuh"  // compute_haversine
#include <cuvs/distance/distance.hpp>

#include <cstdint>  // uint32_t

namespace cuvs::neighbors::ball_cover::detail {

template <typename value_t, typename value_int>
struct DistFunc {
  virtual __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                                 const value_t* b,
                                                                 const value_int n_dims)
  {
    return -1;
  };
};

template <typename value_t, typename value_int>
struct HaversineFunc : public DistFunc<value_t, value_int> {
  __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                         const value_t* b,
                                                         const value_int n_dims) override
  {
    return cuvs::neighbors::detail::compute_haversine<value_t, value_t>(a[0], b[0], a[1], b[1]);
  }
};

template <typename value_t, typename value_int>
struct EuclideanFunc : public DistFunc<value_t, value_int> {
  __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                         const value_t* b,
                                                         const value_int n_dims) override
  {
    value_t sum_sq = 0;
    for (value_int i = 0; i < n_dims; ++i) {
      value_t diff = a[i] - b[i];
      sum_sq += diff * diff;
    }

    return raft::sqrt(sum_sq);
  }
};

template <typename value_t, typename value_int>
struct EuclideanSqFunc : public DistFunc<value_t, value_int> {
  __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                         const value_t* b,
                                                         const value_int n_dims) override
  {
    value_t sum_sq = 0;
    for (value_int i = 0; i < n_dims; ++i) {
      value_t diff = a[i] - b[i];
      sum_sq += diff * diff;
    }
    return sum_sq;
  }
};

// Direct distance function for use in kernels that need metric information
template <typename value_t, typename value_int>
__device__ __host__ __forceinline__ value_t compute_distance_by_metric(
  const value_t* a, const value_t* b, const value_int n_dims, cuvs::distance::DistanceType metric)
{
  if (metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      metric == cuvs::distance::DistanceType::L2SqrtUnexpanded) {
    // Euclidean distance implementation
    return EuclideanFunc<value_t, value_int>{}(a, b, n_dims);
  } else if (metric == cuvs::distance::DistanceType::Haversine) {
    // Haversine distance implementation
    return HaversineFunc<value_t, value_int>{}(a, b, n_dims);
  }

  // Default case
  return -1;
}

};  // namespace cuvs::neighbors::ball_cover::detail
