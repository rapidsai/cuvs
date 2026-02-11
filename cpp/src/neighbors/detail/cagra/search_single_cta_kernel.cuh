/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_distance-ext.cuh"

#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

template <typename DataT,
          typename index_t,
          typename distance_t,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(
  const dataset_descriptor_host<DataT, index_t, distance_t>& dataset_desc,
  raft::device_matrix_view<const index_t, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
  uintptr_t topk_indices_ptr,      // [num_queries, topk]
  distance_t* topk_distances_ptr,  // [num_queries, topk]
  const DataT* queries_ptr,        // [num_queries, dataset_dim]
  uint32_t num_queries,
  const index_t* dev_seed_ptr,        // [num_queries, num_seeds]
  uint32_t* num_executed_iterations,  // [num_queries,]
  const search_params& ps,
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,  //
  uint32_t smem_size,
  int64_t hash_bitlen,
  index_t* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_seeds,
  SampleFilterT sample_filter,
  cudaStream_t stream);

}
