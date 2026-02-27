/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_vpq-impl.cuh"
#include "../device_common.hpp"  // For dataset_descriptor_base_t

namespace cuvs::neighbors::cagra::detail {

// Unified compute_distance implementation for VPQ descriptors
// This is instantiated when PQ_BITS>0, PQ_LEN>0, CodebookT=half
// QueryT is always half for VPQ
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
__device__ DistanceT
compute_distance(const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                 IndexT dataset_index)
{
  // For VPQ descriptors, PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half
  static_assert(
    PQ_BITS > 0 && PQ_LEN > 0 && std::is_same_v<CodebookT, half> && std::is_same_v<QueryT, half>,
    "VPQ descriptor requires PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half");

  // Reconstruct the descriptor type and call compute_distance_vpq
  // QueryT is always half for VPQ
  using desc_t = cagra_q_dataset_descriptor_t<TeamSize,
                                              DatasetBlockDim,
                                              PQ_BITS,
                                              PQ_LEN,
                                              CodebookT,
                                              DataT,
                                              IndexT,
                                              DistanceT,
                                              QueryT>;
  return compute_distance_vpq<desc_t>(args, dataset_index);
}

}  // namespace cuvs::neighbors::cagra::detail
