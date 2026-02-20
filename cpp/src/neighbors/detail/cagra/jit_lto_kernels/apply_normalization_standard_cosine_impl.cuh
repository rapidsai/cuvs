/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_standard-impl.cuh"

namespace cuvs::neighbors::cagra::detail {

// Cosine normalization fragment implementation
// This provides apply_normalization_standard that normalizes by dataset norm (for CosineExpanded
// metric)
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
  // CosineExpanded normalization: divide by dataset norm
  const auto* dataset_norms =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>::
      dataset_norms_ptr(args);
  auto norm = dataset_norms[dataset_index];
  if (norm > 0) { distance = distance / norm; }
  return distance;
}

}  // namespace cuvs::neighbors::cagra::detail
