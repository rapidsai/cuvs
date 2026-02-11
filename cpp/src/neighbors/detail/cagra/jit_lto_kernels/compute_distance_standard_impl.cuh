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

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Extern function implementation for compute_distance_standard (standard descriptor)
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ DistanceT compute_distance_standard(const DataT* dataset_ptr,
                                               uint32_t smem_ws_ptr,
                                               IndexT dataset_index,
                                               uint32_t dim,
                                               uint32_t ld,
                                               uint32_t team_size_bitshift,
                                               const DistanceT* dataset_norms)
{
  using desc_type = cuvs::neighbors::cagra::detail::
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  using base_type = typename desc_type::base_type;
  using args_t    = typename base_type::args_t;

  // Reconstruct args_t from parameters
  args_t args;
  args.smem_ws_ptr = smem_ws_ptr;
  args.dim         = dim;
  args.extra_word1 = ld;                    // dataset_ld
  args.extra_ptr1  = (void*)dataset_ptr;    // dataset_ptr
  args.extra_ptr2  = (void*)dataset_norms;  // dataset_norms

  // Call the free function compute_distance_standard
  auto per_thread_distances =
    cuvs::neighbors::cagra::detail::compute_distance_standard<desc_type>(args, dataset_index);

  // Use team_sum with the provided team_size_bitshift
  return device::team_sum(per_thread_distances, team_size_bitshift);
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
