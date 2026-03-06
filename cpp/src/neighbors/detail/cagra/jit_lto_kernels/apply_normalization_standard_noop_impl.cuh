/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_standard-impl.cuh"

namespace cuvs::neighbors::cagra::detail {

// No-op normalization fragment implementation
// This provides apply_normalization_standard that does nothing (for non-CosineExpanded metrics)
// QueryT is needed to match the descriptor template signature, but not used in this function
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
__device__ DistanceT
apply_normalization_standard(DistanceT distance,
                             const typename cuvs::neighbors::cagra::detail::
                               dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                             IndexT dataset_index)
{
  // No normalization needed for non-CosineExpanded metrics
  return distance;
}

}  // namespace cuvs::neighbors::cagra::detail
