/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_standard-impl.cuh"
#include "../device_common.hpp"

#include <raft/core/operators.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

namespace cuvs::neighbors::cagra::detail {

// Extern function implementation for compute_distance_standard (standard descriptor)
// Returns per-thread distance (team_sum must be called by the caller)
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ DistanceT
compute_distance_standard(const typename cuvs::neighbors::cagra::detail::
                            dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                          IndexT dataset_index)
{
  // Call the free function compute_distance_standard directly with args (already loaded)
  // Returns per-thread distance (caller must do team_sum)
  using desc_t = cuvs::neighbors::cagra::detail::
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  auto per_thread_distance =
    cuvs::neighbors::cagra::detail::compute_distance_standard<desc_t>(args, dataset_index);
  return per_thread_distance;
}

}  // namespace cuvs::neighbors::cagra::detail
