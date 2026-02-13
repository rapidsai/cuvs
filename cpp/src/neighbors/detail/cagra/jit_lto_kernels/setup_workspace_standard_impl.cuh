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

// Extern function implementation for setup_workspace_standard (standard descriptor)
// Takes the concrete descriptor pointer and calls the free function directly (not through function
// pointer) For JIT LTO, the descriptor's setup_workspace_impl is nullptr, so we must call the free
// function directly
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ const cuvs::neighbors::cagra::detail::
  standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>*
  setup_workspace_standard(
    const cuvs::neighbors::cagra::detail::
      standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>*
        desc,
    void* smem,
    const DataT* queries,
    uint32_t query_id)
{
  // CRITICAL: This function uses __syncthreads() and expects ALL threads to call it
  // If only thread 0 calls it, __syncthreads() will hang forever
  // Call the free function directly (not desc->setup_workspace() which uses a function pointer)
  // The free function is in compute_distance_standard-impl.cuh
  using desc_t = cuvs::neighbors::cagra::detail::
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  const auto* result =
    cuvs::neighbors::cagra::detail::setup_workspace_standard<desc_t>(desc, smem, queries, query_id);
  return result;
}

}  // namespace cuvs::neighbors::cagra::detail
