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

// Extern function implementation for setup_workspace_vpq (VPQ descriptor)
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ uint32_t setup_workspace_vpq(void* smem,
                                        const DataT* queries,
                                        uint32_t query_id,
                                        const uint8_t* encoded_dataset_ptr,
                                        uint32_t encoded_dataset_dim,
                                        const CodebookT* vq_code_book_ptr,
                                        const CodebookT* pq_code_book_ptr,
                                        IndexT dataset_size,
                                        uint32_t dim)
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

  // Create a temporary descriptor on the stack
  desc_type temp_desc(reinterpret_cast<typename desc_type::setup_workspace_type*>(
                        &cuvs::neighbors::cagra::detail::setup_workspace_vpq<desc_type>),
                      reinterpret_cast<typename desc_type::compute_distance_type*>(
                        &cuvs::neighbors::cagra::detail::compute_distance_vpq<desc_type>),
                      encoded_dataset_ptr,
                      encoded_dataset_dim,
                      vq_code_book_ptr,
                      pq_code_book_ptr,
                      dataset_size,
                      dim);

  // Call the free function setup_workspace_vpq which copies descriptor to smem
  const desc_type* result = cuvs::neighbors::cagra::detail::setup_workspace_vpq<desc_type>(
    &temp_desc, smem, queries, query_id);

  // Return the smem_ws_ptr from the descriptor's args
  return result->args.smem_ws_ptr;
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
