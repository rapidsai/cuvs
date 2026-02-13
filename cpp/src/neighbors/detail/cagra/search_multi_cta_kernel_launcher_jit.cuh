/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUVS_ENABLE_JIT_LTO
#error "search_multi_cta_kernel_launcher_jit.cuh included but CUVS_ENABLE_JIT_LTO not defined!"
#endif

// Include tags header before any other includes that might open namespaces
#include <cuvs/detail/jit_lto/cagra/search_single_cta_tags.hpp>

#include "compute_distance.hpp"  // For dataset_descriptor_host
#include "jit_lto_kernels/search_multi_cta_planner.hpp"
#include "search_plan.cuh"      // For search_params
#include "set_value_batch.cuh"  // For set_value_batch
#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>

#include <cuda_runtime.h>
#include <string>
#include <type_traits>
// Note: We don't include search_multi_cta_kernel_jit.cuh here because:
// - The launcher doesn't need the kernel function definitions
// - The kernel is dispatched via the JIT LTO launcher system
// - Including it would pull in impl files that cause namespace issues

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

// Helper functions to get tags for JIT LTO
namespace {
template <typename T>
constexpr auto get_data_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return cuvs::neighbors::cagra::detail::tag_f{}; }
  if constexpr (std::is_same_v<T, __half>) { return cuvs::neighbors::cagra::detail::tag_h{}; }
  if constexpr (std::is_same_v<T, int8_t>) { return cuvs::neighbors::cagra::detail::tag_sc{}; }
  if constexpr (std::is_same_v<T, uint8_t>) { return cuvs::neighbors::cagra::detail::tag_uc{}; }
}

template <typename T>
constexpr auto get_index_type_tag()
{
  if constexpr (std::is_same_v<T, uint32_t>) {
    return cuvs::neighbors::cagra::detail::tag_idx_ui{};
  }
}

template <typename T>
constexpr auto get_distance_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return cuvs::neighbors::cagra::detail::tag_dist_f{}; }
}

template <typename T>
constexpr auto get_source_index_type_tag()
{
  if constexpr (std::is_same_v<T, uint32_t>) {
    return cuvs::neighbors::cagra::detail::tag_idx_ui{};
  }
}

template <class SAMPLE_FILTER_T>
std::string get_sample_filter_name()
{
  if constexpr (std::is_same_v<SAMPLE_FILTER_T, cuvs::neighbors::filtering::none_sample_filter>) {
    return "filter_none";
  } else if constexpr (
    std::is_same_v<SAMPLE_FILTER_T,
                   cuvs::neighbors::filtering::bitset_filter<uint32_t, uint32_t>> ||
    std::is_same_v<SAMPLE_FILTER_T, cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>>) {
    return "filter_bitset";
  } else {
    // Default to none filter for unknown types
    return "filter_none";
  }
}
}  // namespace

// JIT version of select_and_run for multi_cta
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run_jit(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
  const SourceIndexT* source_indices_ptr,
  IndexT* topk_indices_ptr,       // [num_queries, num_cta_per_query, itopk_size]
  DistanceT* topk_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  const DataT* queries_ptr,       // [num_queries, dataset_dim]
  uint32_t num_queries,
  const IndexT* dev_seed_ptr,         // [num_queries, num_seeds]
  uint32_t* num_executed_iterations,  // [num_queries,]
  const search_params& ps,
  uint32_t topk,
  // multi_cta_search (params struct)
  uint32_t block_size,  //
  uint32_t result_buffer_size,
  uint32_t smem_size,
  uint32_t visited_hash_bitlen,
  int64_t traversed_hash_bitlen,
  IndexT* traversed_hashmap_ptr,
  uint32_t num_cta_per_query,
  uint32_t num_seeds,
  SampleFilterT sample_filter,
  cudaStream_t stream)
{
  std::cerr << "[JIT] select_and_run_jit (multi_cta) called (num_queries=" << num_queries
            << ", topk=" << topk << ", num_cta_per_query=" << num_cta_per_query << ")" << std::endl;
  std::cerr.flush();

  // Extract bitset data from filter object (if it's a bitset_filter)
  uint32_t* bitset_ptr        = nullptr;
  SourceIndexT bitset_len     = 0;
  SourceIndexT original_nbits = 0;

  if constexpr (!std::is_same_v<SampleFilterT, cuvs::neighbors::filtering::none_sample_filter>) {
    // Try to extract bitset data from the filter
    if constexpr (std::is_same_v<
                    SampleFilterT,
                    cuvs::neighbors::filtering::bitset_filter<uint32_t, SourceIndexT>>) {
      auto bitset_view = sample_filter.view();
      bitset_ptr       = const_cast<uint32_t*>(bitset_view.data());
      bitset_len       = static_cast<SourceIndexT>(bitset_view.size());
      original_nbits   = static_cast<SourceIndexT>(bitset_view.get_original_nbits());
    }
  }

  // Create planner with tags
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  std::cerr << "[JIT] Using JIT path for CAGRA multi_cta search" << std::endl;
  std::cerr.flush();

  // For multi_cta, we don't use topk_by_bitonic_sort or bitonic_sort_and_merge_multi_warps
  // These are handled inside the kernel based on max_elements
  // We need to construct the entrypoint name manually since it's different from single_cta
  std::string metric_name_full;
  if (dataset_desc.metric == cuvs::distance::DistanceType::L2Expanded) {
    metric_name_full = "L2Expanded";
  } else if (dataset_desc.metric == cuvs::distance::DistanceType::InnerProduct) {
    metric_name_full = "InnerProduct";
  } else if (dataset_desc.metric == cuvs::distance::DistanceType::CosineExpanded) {
    metric_name_full = "CosineExpanded";
  } else {
    RAFT_FAIL("Unsupported metric for multi_cta JIT kernel");
  }

  // Debug: Check if this is VPQ
  std::cerr << "[JIT] Dataset descriptor - is_vpq: " << dataset_desc.is_vpq
            << ", pq_bits: " << dataset_desc.pq_bits << ", pq_len: " << dataset_desc.pq_len
            << std::endl;
  std::cerr.flush();

  // Create planner and register device functions
  // Pass team_size, dataset_block_dim, and VPQ parameters to match the kernel entrypoint name
  CagraMultiCtaSearchPlanner<DataTag, IndexTag, DistTag, SourceTag> planner(
    dataset_desc.metric,
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
  planner.add_sample_filter_device_function(get_sample_filter_name<SampleFilterT>());

  // Get launcher using the planner's entrypoint name and fragment key
  auto params   = make_fragment_key<DataTag, IndexTag, DistTag, SourceTag>();
  auto launcher = planner.get_launcher();

  uint32_t max_elements{};
  if (result_buffer_size <= 64) {
    max_elements = 64;
  } else if (result_buffer_size <= 128) {
    max_elements = 128;
  } else if (result_buffer_size <= 256) {
    max_elements = 256;
  } else {
    THROW("Result buffer size %u larger than max buffer size %u", result_buffer_size, 256);
  }

  RAFT_CUDA_TRY(cudaFuncSetAttribute(
    launcher->get_kernel(), cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  // Initialize hash table
  const uint32_t traversed_hash_size = hashmap::get_size(traversed_hash_bitlen);
  set_value_batch(traversed_hashmap_ptr,
                  traversed_hash_size,
                  ~static_cast<IndexT>(0),
                  traversed_hash_size,
                  num_queries,
                  stream);

  dim3 block_dims(block_size, 1, 1);
  dim3 grid_dims(num_cta_per_query, num_queries, 1);
  RAFT_LOG_DEBUG("Launching JIT multi_cta kernel with %u threads, (%u, %u) blocks %u smem",
                 block_size,
                 num_cta_per_query,
                 num_queries,
                 smem_size);

  // Get the device descriptor pointer
  // dev_ptr() returns const dataset_descriptor_base_t*, but kernel expects const desc_t*
  // Since base class is at offset 0, pointer value is the same and kernel can safely cast
  const auto* dev_desc = dataset_desc.dev_ptr(stream);

  // Dispatch kernel via launcher
  launcher->dispatch(stream,
                     grid_dims,
                     block_dims,
                     smem_size,
                     topk_indices_ptr,
                     topk_distances_ptr,
                     dev_desc,
                     queries_ptr,
                     graph.data_handle(),
                     max_elements,
                     graph.extent(1),
                     source_indices_ptr,
                     ps.num_random_samplings,
                     ps.rand_xor_mask,
                     dev_seed_ptr,
                     num_seeds,
                     visited_hash_bitlen,
                     traversed_hashmap_ptr,
                     traversed_hash_bitlen,
                     ps.itopk_size,
                     ps.min_iterations,
                     ps.max_iterations,
                     num_executed_iterations,
                     bitset_ptr,
                     bitset_len,
                     original_nbits);

  // Check for errors immediately after launch
  cudaError_t err = cudaPeekAtLastError();
  if (err != cudaSuccess) {
    std::cerr << "[JIT] ERROR after kernel launch (peek): " << cudaGetErrorString(err) << " ("
              << err << ")" << std::endl;
    std::cerr.flush();
  } else {
    std::cerr << "[JIT] No error after kernel launch (peek)" << std::endl;
    std::cerr.flush();
  }
  RAFT_CUDA_TRY(err);

  // Synchronize and check again - this will catch kernel execution errors
  std::cerr << "[JIT] Synchronizing stream to check for kernel execution errors..." << std::endl;
  std::cerr.flush();
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "[JIT] ERROR after kernel sync: " << cudaGetErrorString(err) << " (" << err << ")"
              << std::endl;
    std::cerr.flush();
  } else {
    std::cerr << "[JIT] Stream synchronized successfully - kernel completed" << std::endl;
    std::cerr.flush();
  }
  RAFT_CUDA_TRY(err);
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
