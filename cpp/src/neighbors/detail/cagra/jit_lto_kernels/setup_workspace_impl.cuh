/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../device_common.hpp"

namespace cuvs::neighbors::cagra::detail {

// Unified setup_workspace function - takes dataset_descriptor_base_t* and template parameters
// Standard and VPQ versions are in separate impl headers but use the same function name
// The planner links the appropriate fragment at runtime based on PQ_BITS/PQ_LEN
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ dataset_descriptor_base_t<DataT, IndexT, DistanceT>* setup_workspace(
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* desc_ptr,
  void* smem,
  const DataT* queries,
  uint32_t query_id);

}  // namespace cuvs::neighbors::cagra::detail
