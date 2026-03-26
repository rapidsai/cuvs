/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::ivf_pq::detail {

template <bool PrecompBaseDiff>
__device__ void precompute_base_diff_impl(cuvs::distance::DistanceType metric,
                                          uint32_t dim,
                                          uint8_t* lut_end,
                                          const float* query,
                                          const float* cluster_center)
{
  if constexpr (PrecompBaseDiff) {
    // Reduce number of memory reads later by pre-computing parts of the score
    switch (metric) {
      case distance::DistanceType::L2SqrtExpanded:
      case distance::DistanceType::L2Expanded: {
        for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
          reinterpret_cast<float*>(lut_end)[i] = query[i] - cluster_center[i];
        }
      } break;
      case distance::DistanceType::CosineExpanded:
      case distance::DistanceType::InnerProduct: {
        float2 pvals;
        for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
          pvals.x                               = query[i];
          pvals.y                               = cluster_center[i] * pvals.x;
          reinterpret_cast<float2*>(lut_end)[i] = pvals;
        }
      } break;
      default: __builtin_unreachable();
    }
    __syncthreads();
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
