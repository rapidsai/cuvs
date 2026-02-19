/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_vpq-impl.cuh"
#include "../device_common.hpp"  // For dataset_descriptor_base_t
#include "compute_distance_impl.cuh"

namespace cuvs::neighbors::cagra::detail {

// Unified compute_distance implementation for VPQ descriptors
// This is instantiated when PQ_BITS>0, PQ_LEN>0, CodebookT=half
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ DistanceT
compute_distance(const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                 IndexT dataset_index)
{
  // For VPQ descriptors, PQ_BITS>0, PQ_LEN>0, CodebookT=half
  static_assert(PQ_BITS > 0 && PQ_LEN > 0 && std::is_same_v<CodebookT, half>,
                "VPQ descriptor requires PQ_BITS>0, PQ_LEN>0, CodebookT=half");

  // Reconstruct the descriptor type and call compute_distance_vpq
  using desc_t = cagra_q_dataset_descriptor_t<TeamSize,
                                              DatasetBlockDim,
                                              PQ_BITS,
                                              PQ_LEN,
                                              CodebookT,
                                              DataT,
                                              IndexT,
                                              DistanceT>;
  return compute_distance_vpq<desc_t>(args, dataset_index);
}

}  // namespace cuvs::neighbors::cagra::detail
