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

// Extern function implementation for setup_workspace_vpq (VPQ descriptor)
// Takes the concrete descriptor pointer and calls the free function directly (not through function
// pointer) For JIT LTO, the descriptor's setup_workspace_impl is nullptr, so we must call the free
// function directly
// Note: Metric is no longer a template parameter - VPQ only supports L2Expanded
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ const cuvs::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<TeamSize,
                                                                              DatasetBlockDim,
                                                                              PQ_BITS,
                                                                              PQ_LEN,
                                                                              CodebookT,
                                                                              DataT,
                                                                              IndexT,
                                                                              DistanceT>*
setup_workspace_vpq(
  const cuvs::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<TeamSize,
                                                                     DatasetBlockDim,
                                                                     PQ_BITS,
                                                                     PQ_LEN,
                                                                     CodebookT,
                                                                     DataT,
                                                                     IndexT,
                                                                     DistanceT>* desc,
  void* smem,
  const DataT* queries,
  uint32_t query_id)
{
  // Call the free function directly (not desc->setup_workspace() which uses a function pointer)
  // The free function is in compute_distance_vpq-impl.cuh
  // VPQ only supports L2Expanded, so Metric is hardcoded
  using desc_t = cuvs::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<TeamSize,
                                                                              DatasetBlockDim,
                                                                              PQ_BITS,
                                                                              PQ_LEN,
                                                                              CodebookT,
                                                                              DataT,
                                                                              IndexT,
                                                                              DistanceT>;
  const auto* result =
    cuvs::neighbors::cagra::detail::setup_workspace_vpq<desc_t>(desc, smem, queries, query_id);
  return result;
}

}  // namespace cuvs::neighbors::cagra::detail
