/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_vpq-impl.cuh"
#include "../device_common.hpp"

namespace cuvs::neighbors::cagra::detail {

// Unified setup_workspace implementation for VPQ descriptors
// This is instantiated when PQ_BITS>0, PQ_LEN>0, CodebookT=half
// Takes dataset_descriptor_base_t* and reconstructs the derived descriptor inside
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
__device__ dataset_descriptor_base_t<DataT, IndexT, DistanceT>* setup_workspace(
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* desc_ptr,
  void* smem,
  const DataT* queries,
  uint32_t query_id)
{
  // For VPQ descriptors, PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half
  static_assert(
    PQ_BITS > 0 && PQ_LEN > 0 && std::is_same_v<CodebookT, half> && std::is_same_v<QueryT, half>,
    "VPQ descriptor requires PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half");

  // Reconstruct the descriptor pointer from base pointer
  // QueryT is always half for VPQ
  using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                    DatasetBlockDim,
                                                    PQ_BITS,
                                                    PQ_LEN,
                                                    CodebookT,
                                                    DataT,
                                                    IndexT,
                                                    DistanceT,
                                                    QueryT>;
  const desc_t* desc = static_cast<const desc_t*>(desc_ptr);

  // Call the free function directly - it takes DescriptorT as template parameter
  const desc_t* result = setup_workspace_vpq<desc_t>(desc, smem, queries, query_id);
  return const_cast<dataset_descriptor_base_t<DataT, IndexT, DistanceT>*>(
    static_cast<const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*>(result));
}

}  // namespace cuvs::neighbors::cagra::detail
