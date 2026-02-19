/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_vpq-impl.cuh"
#include "../device_common.hpp"

#include <raft/core/operators.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

namespace cuvs::neighbors::cagra::detail {

// Extern function implementation for compute_distance_vpq (VPQ descriptor)
// Returns per-thread distance (team_sum must be called by the caller)
// Note: Metric is no longer a template parameter - VPQ only supports L2Expanded
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ DistanceT
compute_distance_vpq(const typename cuvs::neighbors::cagra::detail::
                       dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                     IndexT dataset_index)
{
  // Call the free function compute_distance_vpq directly with args (already loaded)
  // Returns per-thread distance (caller must do team_sum)
  // VPQ only supports L2Expanded, so Metric is hardcoded
  using desc_t = cuvs::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<TeamSize,
                                                                              DatasetBlockDim,
                                                                              PQ_BITS,
                                                                              PQ_LEN,
                                                                              CodebookT,
                                                                              DataT,
                                                                              IndexT,
                                                                              DistanceT>;
  auto per_thread_distance =
    cuvs::neighbors::cagra::detail::compute_distance_vpq<desc_t>(args, dataset_index);
  return per_thread_distance;
}

}  // namespace cuvs::neighbors::cagra::detail
