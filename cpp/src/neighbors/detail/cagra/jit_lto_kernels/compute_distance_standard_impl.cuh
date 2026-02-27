/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_standard-impl.cuh"
#include "../device_common.hpp"  // For dataset_descriptor_base_t

namespace cuvs::neighbors::cagra::detail {

// Unified compute_distance implementation for standard descriptors
// This is instantiated when PQ_BITS=0, PQ_LEN=0, CodebookT=void
// QueryT can be float (for most metrics) or uint8_t (for BitwiseHamming)
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
  // For standard descriptors, PQ_BITS=0, PQ_LEN=0, CodebookT=void
  static_assert(PQ_BITS == 0 && PQ_LEN == 0 && std::is_same_v<CodebookT, void>,
                "Standard descriptor requires PQ_BITS=0, PQ_LEN=0, CodebookT=void");

  // Reconstruct the descriptor type with QueryT and call compute_distance_standard
  using desc_t =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>;
  return compute_distance_standard<desc_t>(args, dataset_index);
}

}  // namespace cuvs::neighbors::cagra::detail
