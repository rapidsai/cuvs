/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../smem_utils.cuh"

#include "search_single_cta_kernel-inl.cuh"  // For search_kernel_config, persistent_runner_t, etc.
#include "search_single_cta_kernel_launcher_common.cuh"

namespace cuvs::neighbors::cagra::detail::single_cta_search {

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
  cudaStream_t stream)
{
  const SourceIndexT* source_indices_ptr =
    source_indices.has_value() ? source_indices->data_handle() : nullptr;

  // Use common logic to compute launch config
  auto config             = compute_launch_config(num_itopk_candidates, ps.itopk_size, block_size);
  uint32_t max_candidates = config.max_candidates;
  uint32_t max_itopk      = config.max_itopk;

  if (ps.persistent) {
    using runner_type = persistent_runner_t<DataT, IndexT, DistanceT, SourceIndexT, SampleFilterT>;

    get_runner<runner_type>(/*
Note, we're passing the descriptor by reference here, and this reference is going to be passed to a
new spawned thread, which is dangerous. However, the descriptor is copied in that thread before the
control is returned in this thread (in persistent_runner_t constructor), so we're safe.
*/
                            std::cref(dataset_desc),
                            graph,
                            source_indices_ptr,
                            max_candidates,
                            num_itopk_candidates,
                            block_size,
                            smem_size,
                            hash_bitlen,
                            small_hash_bitlen,
                            small_hash_reset_interval,
                            ps.num_random_samplings,
                            ps.rand_xor_mask,
                            num_seeds,
                            max_itopk,
                            ps.itopk_size,
                            ps.search_width,
                            ps.min_iterations,
                            ps.max_iterations,
                            sample_filter,
                            ps.persistent_lifetime,
                            ps.persistent_device_usage)
      ->launch(topk_indices_ptr, topk_distances_ptr, queries_ptr, num_queries, topk);
  } else {
    using descriptor_base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
    auto kernel = search_kernel_config<false, descriptor_base_type, SourceIndexT, SampleFilterT>::
      choose_itopk_and_mx_candidates(ps.itopk_size, num_itopk_candidates, block_size);

    dim3 thread_dims(block_size, 1, 1);
    dim3 block_dims(1, num_queries, 1);
    RAFT_LOG_DEBUG(
      "Launching kernel with %u threads, %u block %u smem", block_size, num_queries, smem_size);
    auto const& kernel_launcher = [&](auto const& kernel) -> void {
      kernel<<<block_dims, thread_dims, smem_size, stream>>>(topk_indices_ptr,
                                                             topk_distances_ptr,
                                                             topk,
                                                             dataset_desc.dev_ptr(stream),
                                                             queries_ptr,
                                                             graph.data_handle(),
                                                             graph.extent(1),
                                                             source_indices_ptr,
                                                             ps.num_random_samplings,
                                                             ps.rand_xor_mask,
                                                             dev_seed_ptr,
                                                             num_seeds,
                                                             hashmap_ptr,
                                                             max_candidates,
                                                             max_itopk,
                                                             ps.itopk_size,
                                                             ps.search_width,
                                                             ps.min_iterations,
                                                             ps.max_iterations,
                                                             num_executed_iterations,
                                                             hash_bitlen,
                                                             small_hash_bitlen,
                                                             small_hash_reset_interval,
                                                             sample_filter);
    };
    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
      kernel, smem_size, kernel_launcher);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
