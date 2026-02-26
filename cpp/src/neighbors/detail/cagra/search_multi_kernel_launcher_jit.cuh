/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUVS_ENABLE_JIT_LTO
#error "search_multi_kernel_launcher_jit.cuh included but CUVS_ENABLE_JIT_LTO not defined!"
#endif

// Tags header should be included before this header (at file scope, not inside functions)
// to avoid namespace definition errors when this header is included inside function bodies

#include "compute_distance.hpp"  // For dataset_descriptor_host
#include "jit_lto_kernels/search_multi_kernel_planner.hpp"
#include "sample_filter_utils.cuh"  // For CagraSampleFilterWithQueryIdOffset
#include "search_plan.cuh"          // For search_params
#include "shared_launcher_jit.hpp"  // For shared JIT helper functions
#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>

#include <cuda_runtime.h>
#include <string>
#include <type_traits>
// - The launcher doesn't need the kernel function definitions
// - The kernel is dispatched via the JIT LTO launcher system
// - Including it would pull in impl files that cause namespace issues

namespace cuvs::neighbors::cagra::detail::multi_kernel_search {

// JIT version of random_pickup
template <typename DataT, typename IndexT, typename DistanceT>
void random_pickup_jit(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                       const DataT* queries_ptr,  // [num_queries, dataset_dim]
                       std::size_t num_queries,
                       std::size_t num_pickup,
                       unsigned num_distilation,
                       uint64_t rand_xor_mask,
                       const IndexT* seed_ptr,  // [num_queries, num_seeds]
                       uint32_t num_seeds,
                       IndexT* result_indices_ptr,       // [num_queries, ldr]
                       DistanceT* result_distances_ptr,  // [num_queries, ldr]
                       std::size_t ldr,                  // (*) ldr >= num_pickup
                       IndexT* visited_hashmap_ptr,      // [num_queries, 1 << bitlen]
                       std::uint32_t hash_bitlen,
                       cudaStream_t cuda_stream)
{
  // Create planner with tags
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<IndexT>());  // Use IndexT for source

  // Create planner and register device functions
  std::shared_ptr<AlgorithmLauncher> launcher;
  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
      planner(dataset_desc.metric,
              "random_pickup",
              dataset_desc.team_size,
              dataset_desc.dataset_block_dim,
              dataset_desc.is_vpq,
              dataset_desc.pq_bits,
              dataset_desc.pq_len);
    planner.add_setup_workspace_device_function(dataset_desc.metric,
                                                dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim,
                                                dataset_desc.is_vpq,
                                                dataset_desc.pq_bits,
                                                dataset_desc.pq_len);
    planner.add_compute_distance_device_function(dataset_desc.metric,
                                                 dataset_desc.team_size,
                                                 dataset_desc.dataset_block_dim,
                                                 dataset_desc.is_vpq,
                                                 dataset_desc.pq_bits,
                                                 dataset_desc.pq_len);
    launcher = planner.get_launcher();
  } else {
    using CodebookTag = codebook_tag_standard_t;
    if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
      using QueryTag =
        query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
      CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                "random_pickup",
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);
      planner.add_setup_workspace_device_function(dataset_desc.metric,
                                                  dataset_desc.team_size,
                                                  dataset_desc.dataset_block_dim,
                                                  dataset_desc.is_vpq,
                                                  dataset_desc.pq_bits,
                                                  dataset_desc.pq_len);
      planner.add_compute_distance_device_function(dataset_desc.metric,
                                                   dataset_desc.team_size,
                                                   dataset_desc.dataset_block_dim,
                                                   dataset_desc.is_vpq,
                                                   dataset_desc.pq_bits,
                                                   dataset_desc.pq_len);
      launcher = planner.get_launcher();
    } else {
      using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
      CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                "random_pickup",
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);
      planner.add_setup_workspace_device_function(dataset_desc.metric,
                                                  dataset_desc.team_size,
                                                  dataset_desc.dataset_block_dim,
                                                  dataset_desc.is_vpq,
                                                  dataset_desc.pq_bits,
                                                  dataset_desc.pq_len);
      planner.add_compute_distance_device_function(dataset_desc.metric,
                                                   dataset_desc.team_size,
                                                   dataset_desc.dataset_block_dim,
                                                   dataset_desc.is_vpq,
                                                   dataset_desc.pq_bits,
                                                   dataset_desc.pq_len);
      launcher = planner.get_launcher();
    }
  }

  const auto block_size                = 256u;
  const auto num_teams_per_threadblock = block_size / dataset_desc.team_size;
  const dim3 grid_size((num_pickup + num_teams_per_threadblock - 1) / num_teams_per_threadblock,
                       num_queries);

  // Get the device descriptor pointer
  const auto* dev_desc = dataset_desc.dev_ptr(cuda_stream);

  // Cast size_t parameters to match kernel signature exactly
  // The dispatch mechanism uses void* pointers, so parameter sizes must match exactly
  const uint32_t ldr_u32 = static_cast<uint32_t>(ldr);

  // Dispatch kernel via launcher
  launcher->dispatch(cuda_stream,
                     grid_size,
                     dim3(block_size, 1, 1),
                     dataset_desc.smem_ws_size_in_bytes,
                     dev_desc,
                     queries_ptr,
                     num_pickup,
                     num_distilation,
                     rand_xor_mask,
                     seed_ptr,
                     num_seeds,
                     result_indices_ptr,
                     result_distances_ptr,
                     ldr_u32,  // Cast size_t to uint32_t
                     visited_hashmap_ptr,
                     hash_bitlen);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// JIT version of compute_distance_to_child_nodes
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          class SourceIndexT,
          class SAMPLE_FILTER_T>
void compute_distance_to_child_nodes_jit(
  const IndexT* parent_node_list,        // [num_queries, search_width]
  IndexT* const parent_candidates_ptr,   // [num_queries, search_width]
  DistanceT* const parent_distance_ptr,  // [num_queries, search_width]
  std::size_t lds,
  uint32_t search_width,
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const IndexT* neighbor_graph_ptr,  // [dataset_size, graph_degree]
  std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const DataT* query_ptr,  // [num_queries, data_dim]
  std::uint32_t num_queries,
  IndexT* visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  std::uint32_t hash_bitlen,
  IndexT* result_indices_ptr,       // [num_queries, ldd]
  DistanceT* result_distances_ptr,  // [num_queries, ldd]
  std::uint32_t ldd,                // (*) ldd >= search_width * graph_degree
  SAMPLE_FILTER_T sample_filter,
  cudaStream_t cuda_stream)
{
  // Create planner with tags
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  // Create planner and register device functions
  std::shared_ptr<AlgorithmLauncher> launcher;
  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
      planner(dataset_desc.metric,
              "compute_distance_to_child_nodes",
              dataset_desc.team_size,
              dataset_desc.dataset_block_dim,
              dataset_desc.is_vpq,
              dataset_desc.pq_bits,
              dataset_desc.pq_len);
    planner.add_setup_workspace_device_function(dataset_desc.metric,
                                                dataset_desc.team_size,
                                                dataset_desc.dataset_block_dim,
                                                dataset_desc.is_vpq,
                                                dataset_desc.pq_bits,
                                                dataset_desc.pq_len);
    planner.add_compute_distance_device_function(dataset_desc.metric,
                                                 dataset_desc.team_size,
                                                 dataset_desc.dataset_block_dim,
                                                 dataset_desc.is_vpq,
                                                 dataset_desc.pq_bits,
                                                 dataset_desc.pq_len);
    launcher = planner.get_launcher();
  } else {
    using CodebookTag = codebook_tag_standard_t;
    if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
      using QueryTag =
        query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
      CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                "compute_distance_to_child_nodes",
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);
      planner.add_setup_workspace_device_function(dataset_desc.metric,
                                                  dataset_desc.team_size,
                                                  dataset_desc.dataset_block_dim,
                                                  dataset_desc.is_vpq,
                                                  dataset_desc.pq_bits,
                                                  dataset_desc.pq_len);
      planner.add_compute_distance_device_function(dataset_desc.metric,
                                                   dataset_desc.team_size,
                                                   dataset_desc.dataset_block_dim,
                                                   dataset_desc.is_vpq,
                                                   dataset_desc.pq_bits,
                                                   dataset_desc.pq_len);
      launcher = planner.get_launcher();
    } else {
      using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
      CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                "compute_distance_to_child_nodes",
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);
      planner.add_setup_workspace_device_function(dataset_desc.metric,
                                                  dataset_desc.team_size,
                                                  dataset_desc.dataset_block_dim,
                                                  dataset_desc.is_vpq,
                                                  dataset_desc.pq_bits,
                                                  dataset_desc.pq_len);
      planner.add_compute_distance_device_function(dataset_desc.metric,
                                                   dataset_desc.team_size,
                                                   dataset_desc.dataset_block_dim,
                                                   dataset_desc.is_vpq,
                                                   dataset_desc.pq_bits,
                                                   dataset_desc.pq_len);
      launcher = planner.get_launcher();
    }
  }

  const auto block_size      = 128;
  const auto teams_per_block = block_size / dataset_desc.team_size;
  const dim3 grid_size((search_width * graph_degree + teams_per_block - 1) / teams_per_block,
                       num_queries);

  // Get the device descriptor pointer
  const auto* dev_desc = dataset_desc.dev_ptr(cuda_stream);

  // Dispatch kernel via launcher
  launcher->dispatch(cuda_stream,
                     grid_size,
                     dim3(block_size, 1, 1),
                     dataset_desc.smem_ws_size_in_bytes,
                     parent_node_list,
                     parent_candidates_ptr,
                     parent_distance_ptr,
                     lds,
                     search_width,
                     dev_desc,
                     neighbor_graph_ptr,
                     graph_degree,
                     source_indices_ptr,
                     query_ptr,
                     visited_hashmap_ptr,
                     hash_bitlen,
                     result_indices_ptr,
                     result_distances_ptr,
                     ldd,
                     sample_filter);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// JIT version of apply_filter
template <class INDEX_T, class DISTANCE_T, class SourceIndexT, class SAMPLE_FILTER_T>
void apply_filter_jit(const SourceIndexT* source_indices_ptr,
                      INDEX_T* const result_indices_ptr,
                      DISTANCE_T* const result_distances_ptr,
                      const std::size_t lds,
                      const std::uint32_t result_buffer_size,
                      const std::uint32_t num_queries,
                      const INDEX_T query_id_offset,
                      SAMPLE_FILTER_T sample_filter,
                      cudaStream_t cuda_stream)
{
  // Extract bitset data from filter object (if it's a bitset_filter)
  uint32_t* bitset_ptr        = nullptr;
  SourceIndexT bitset_len     = 0;
  SourceIndexT original_nbits = 0;

  // Check if it has the wrapper members (CagraSampleFilterWithQueryIdOffset)
  // Note: query_id_offset is already a parameter to this function, so we don't extract it here
  if constexpr (requires {
                  sample_filter.filter;
                  sample_filter.offset;
                }) {
    using InnerFilter = decltype(sample_filter.filter);
    if constexpr (is_bitset_filter<InnerFilter>::value) {
      // Extract bitset data for bitset_filter (works for any bitset_filter instantiation)
      auto bitset_view = sample_filter.filter.view();
      bitset_ptr       = const_cast<uint32_t*>(bitset_view.data());
      bitset_len       = static_cast<SourceIndexT>(bitset_view.size());
      original_nbits   = static_cast<SourceIndexT>(bitset_view.get_original_nbits());
    }
  }

  // Create planner with tags
  using DataTag =
    decltype(get_data_type_tag<float>());  // Not used for apply_filter, but required by planner
  using IndexTag  = decltype(get_index_type_tag<INDEX_T>());
  using DistTag   = decltype(get_distance_type_tag<DISTANCE_T>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  // Create planner - apply_filter doesn't use dataset_descriptor, so we use dummy values
  // The kernel name is "apply_filter_kernel" and build_entrypoint_name will handle it specially
  using QueryTag    = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  using CodebookTag = void;
  CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
    planner(cuvs::distance::DistanceType::L2Expanded,
            "apply_filter_kernel",
            8,
            128,
            false,
            0,
            0);  // Dummy values, not used by apply_filter

  // Add sample filter device function - determine filter type from template parameter
  planner.add_sample_filter_device_function(get_sample_filter_name<SAMPLE_FILTER_T>());

  auto launcher = planner.get_launcher();

  const std::uint32_t block_size = 256;
  const std::uint32_t grid_size  = raft::ceildiv(num_queries * result_buffer_size, block_size);

  // Dispatch kernel via launcher with bitset parameters
  launcher->dispatch(cuda_stream,
                     dim3(grid_size, 1, 1),
                     dim3(block_size, 1, 1),
                     0,  // No shared memory needed
                     source_indices_ptr,
                     result_indices_ptr,
                     result_distances_ptr,
                     lds,
                     result_buffer_size,
                     num_queries,
                     query_id_offset,
                     bitset_ptr,
                     bitset_len,
                     original_nbits);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::cagra::detail::multi_kernel_search
