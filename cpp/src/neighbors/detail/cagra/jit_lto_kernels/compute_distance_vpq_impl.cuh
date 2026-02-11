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

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Extern function implementation for compute_distance_vpq (VPQ descriptor)
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ DistanceT compute_distance_vpq(const uint8_t* encoded_dataset_ptr,
                                          uint32_t smem_ws_ptr,
                                          IndexT dataset_index,
                                          uint32_t encoded_dataset_dim,
                                          const CodebookT* vq_code_book_ptr,
                                          const CodebookT* pq_code_book_ptr,
                                          uint32_t team_size_bitshift)
{
  using desc_type = cuvs::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<Metric,
                                                                                 TeamSize,
                                                                                 DatasetBlockDim,
                                                                                 PQ_BITS,
                                                                                 PQ_LEN,
                                                                                 CodebookT,
                                                                                 DataT,
                                                                                 IndexT,
                                                                                 DistanceT>;
  using base_type = typename desc_type::base_type;
  using args_t    = typename base_type::args_t;

  // Reconstruct args_t from parameters
  args_t args;
  args.smem_ws_ptr = smem_ws_ptr;
  args.dim         = encoded_dataset_dim;
  args.extra_word1 = encoded_dataset_dim;
  args.extra_ptr1  = (void*)encoded_dataset_ptr;
  args.extra_ptr2  = (void*)vq_code_book_ptr;
  // Note: pq_code_book_ptr is stored in shared memory (copied during setup_workspace_vpq),
  // and compute_distance_vpq accesses it via args.smem_ws_ptr, so we don't need to pass it
  // separately.

  // Call the free function compute_distance_vpq
  // It will access the codebook from shared memory via smem_ws_ptr
  auto per_thread_distances =
    cuvs::neighbors::cagra::detail::compute_distance_vpq<desc_type>(args, dataset_index);

  // Use team_sum with the provided team_size_bitshift
  return device::team_sum(per_thread_distances, team_size_bitshift);
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
