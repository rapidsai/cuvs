/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <neighbors/detail/cagra/compute_distance-ext.cuh>

#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

/**
 * @brief Per-segment descriptor for the multi-segment CAGRA search kernel.
 *
 * One instance per Lucene segment; the kernel reads this array from device memory using
 * blockIdx.z as the segment index.
 */
template <typename DataT, typename IndexT, typename DistanceT>
struct alignas(16) multi_segment_desc_t {
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc;
  const DataT* queries_ptr;         // [num_queries, dim] for this segment
  const IndexT* graph;              // [dataset_size, graph_degree]
  uint32_t graph_degree;
  uint32_t _pad;
  uintptr_t result_indices_ptr;     // tagged pointer: [num_queries, top_k]
  DistanceT* result_distances_ptr;  // [num_queries, top_k]
};

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
  std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
  uintptr_t topk_indices_ptr,     // [num_queries, topk]
  DistanceT* topk_distances_ptr,  // [num_queries, topk]
  const DataT* queries_ptr,       // [num_queries, dataset_dim]
  uint32_t num_queries,
  const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
  uint32_t* num_executed_iterations,  // [num_queries,]
  const search_params& ps,
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,  //
  uint32_t smem_size,
  int64_t hash_bitlen,
  IndexT* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_seeds,
  SampleFilterT sample_filter,
  cudaStream_t stream);

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run_multi_segment(
  const multi_segment_desc_t<DataT, IndexT, DistanceT>* segment_descs,
  uint32_t num_segments,
  uint32_t num_queries,
  const search_params& ps,
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,
  uint32_t smem_size,
  int64_t hash_bitlen,
  IndexT* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  SampleFilterT sample_filter,
  cudaStream_t stream);

}
