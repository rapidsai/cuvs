/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include <cuvs/detail/jit_lto/ivf_pq/compute_similarity_fragments.hpp>
#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename MetricTag>
__device__ void precompute_base_diff_impl(uint32_t dim,
                                          uint8_t* lut_end,
                                          const float* query,
                                          const float* cluster_center)
{
  // Reduce number of memory reads later by pre-computing parts of the score
  if constexpr (std::is_same_v<MetricTag, tag_metric_euclidean>) {
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
      reinterpret_cast<float*>(lut_end)[i] = query[i] - cluster_center[i];
    }
    __syncthreads();
  } else if constexpr (std::is_same_v<MetricTag, tag_metric_inner_product>) {
    float2 pvals;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
      pvals.x                               = query[i];
      pvals.y                               = cluster_center[i] * pvals.x;
      reinterpret_cast<float2*>(lut_end)[i] = pvals;
    }
    __syncthreads();
  } else {
    static_assert(std::is_same_v<MetricTag, tag_metric_none>, "Invalid MetricTag");
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
