/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance.hpp"  // For dataset_descriptor_base_t definition
#include "../device_common.hpp"

namespace cuvs::neighbors::cagra::detail {

// Unified compute_distance function - takes void* args and template parameters
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
extern __device__ DistanceT
compute_distance(const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                 IndexT dataset_index);

}  // namespace cuvs::neighbors::cagra::detail
