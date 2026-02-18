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
#include "sample_filter_utils.cuh"  // For CagraSampleFilterWithQueryIdOffset
#include "search_plan.cuh"          // For search_params
#include "set_value_batch.cuh"      // For set_value_batch
#include "shared_launcher_jit.hpp"  // For shared JIT helper functions
#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>

#include <cuda_runtime.h>
#include <string>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

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
  RAFT_LOG_INFO(
    "[JIT LAUNCHER] Entering MULTI_CTA launcher (num_queries=%u, topk=%u, num_cta_per_query=%u, "
    "result_buffer_size=%u, itopk_size=%zu)",
    num_queries,
    topk,
    num_cta_per_query,
    result_buffer_size,
    ps.itopk_size);
  // Extract bitset data from filter object (if it's a bitset_filter)
  uint32_t* bitset_ptr        = nullptr;
  SourceIndexT bitset_len     = 0;
  SourceIndexT original_nbits = 0;
  uint32_t query_id_offset    = 0;

  // Check if it has the wrapper members (CagraSampleFilterWithQueryIdOffset)
  if constexpr (requires {
                  sample_filter.filter;
                  sample_filter.offset;
                }) {
    using InnerFilter = decltype(sample_filter.filter);
    // Always extract offset for wrapped filters
    query_id_offset = sample_filter.offset;
    RAFT_LOG_INFO("Extracted query_id_offset: %u", query_id_offset);
    if constexpr (is_bitset_filter<InnerFilter>::value) {
      // Extract bitset data for bitset_filter (works for any bitset_filter instantiation)
      auto bitset_view = sample_filter.filter.view();
      bitset_ptr       = const_cast<uint32_t*>(bitset_view.data());
      bitset_len       = static_cast<SourceIndexT>(bitset_view.size());
      original_nbits   = static_cast<SourceIndexT>(bitset_view.get_original_nbits());
      RAFT_LOG_INFO("Extracted bitset data: bitset_ptr=%p, bitset_len=%zu, original_nbits=%zu",
                    bitset_ptr,
                    static_cast<size_t>(bitset_len),
                    static_cast<size_t>(original_nbits));
    } else {
      RAFT_LOG_INFO("InnerFilter is not bitset_filter, skipping bitset extraction");
    }
  } else {
    RAFT_LOG_INFO("Filter does not have wrapper members (.filter/.offset), skipping extraction");
  }

  // Create planner with tags
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

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
  std::string filter_name = get_sample_filter_name<SampleFilterT>();
  planner.add_sample_filter_device_function(filter_name);
  RAFT_LOG_INFO("[JIT LAUNCHER] MULTI_CTA filter name: %s", filter_name.c_str());

  // Get launcher using the planner's entrypoint name and fragment key
  auto params   = make_fragment_key<DataTag, IndexTag, DistTag, SourceTag>();
  auto launcher = planner.get_launcher();

  if (!launcher) { RAFT_FAIL("Failed to get JIT launcher"); }

  RAFT_LOG_INFO("[JIT LAUNCHER] MULTI_CTA launcher obtained (kernel handle: %p)",
                launcher->get_kernel());

  // Verify kernel handle is valid
  cudaKernel_t kernel_handle = launcher->get_kernel();
  if (kernel_handle == nullptr) { RAFT_FAIL("JIT launcher has null kernel handle"); }

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
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_desc_base =
    dataset_desc.dev_ptr(stream);
  const auto* dev_desc = dev_desc_base;
  if (dev_desc == nullptr) { RAFT_FAIL("Device descriptor pointer is NULL"); }

  // Note: dataset_desc is passed by const reference, so it stays alive for the duration of this
  // function The descriptor's state is managed by a shared_ptr internally, so no need to explicitly
  // keep it alive

  // Cast size_t/int64_t parameters to match kernel signature exactly
  // The dispatch mechanism uses void* pointers, so parameter sizes must match exactly
  // graph.extent(1) returns int64_t but kernel expects uint32_t
  // traversed_hash_bitlen is int64_t but kernel expects uint32_t
  // ps.itopk_size, ps.min_iterations, ps.max_iterations are size_t (8 bytes) but kernel expects
  // uint32_t (4 bytes) ps.num_random_samplings is uint32_t but kernel expects unsigned - cast for
  // consistency
  const uint32_t graph_degree_u32          = static_cast<uint32_t>(graph.extent(1));
  const uint32_t traversed_hash_bitlen_u32 = static_cast<uint32_t>(traversed_hash_bitlen);
  const uint32_t itopk_size_u32            = static_cast<uint32_t>(ps.itopk_size);
  const uint32_t min_iterations_u32        = static_cast<uint32_t>(ps.min_iterations);
  const uint32_t max_iterations_u32        = static_cast<uint32_t>(ps.max_iterations);
  const unsigned num_random_samplings_u    = static_cast<unsigned>(ps.num_random_samplings);

  RAFT_LOG_INFO(
    "[JIT LAUNCHER] MULTI_CTA dispatch parameters: graph_degree=%u, traversed_hash_bitlen=%u, "
    "itopk_size=%u, bitset_len=%u, original_nbits=%u, query_id_offset=%u",
    graph_degree_u32,
    traversed_hash_bitlen_u32,
    itopk_size_u32,
    static_cast<uint32_t>(bitset_len),
    static_cast<uint32_t>(original_nbits),
    query_id_offset);

  // Validate critical pointers before dispatch
  if (topk_indices_ptr == nullptr) { RAFT_FAIL("MULTI_CTA: topk_indices_ptr is NULL"); }
  if (topk_distances_ptr == nullptr) { RAFT_FAIL("MULTI_CTA: topk_distances_ptr is NULL"); }
  if (graph.data_handle() == nullptr) { RAFT_FAIL("MULTI_CTA: graph.data_handle() is NULL"); }
  if (dev_desc == nullptr) { RAFT_FAIL("MULTI_CTA: dev_desc is NULL"); }
  RAFT_LOG_INFO(
    "[JIT LAUNCHER] MULTI_CTA pointer validation passed: topk_indices=%p, topk_distances=%p, "
    "graph=%p, dev_desc=%p",
    topk_indices_ptr,
    topk_distances_ptr,
    graph.data_handle(),
    dev_desc);

  // Log all critical parameters before dispatch to help diagnose issues
  RAFT_LOG_INFO(
    "[JIT LAUNCHER] MULTI_CTA pre-dispatch: num_queries=%u, topk=%u, num_cta_per_query=%u, "
    "max_elements=%u, graph.extent(0)=%zu, graph.extent(1)=%zu",
    num_queries,
    topk,
    num_cta_per_query,
    max_elements,
    graph.extent(0),
    graph.extent(1));

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
                     graph_degree_u32,  // Cast int64_t to uint32_t
                     source_indices_ptr,
                     num_random_samplings_u,  // Cast uint32_t to unsigned for consistency
                     ps.rand_xor_mask,        // uint64_t matches kernel (8 bytes)
                     dev_seed_ptr,
                     num_seeds,
                     visited_hash_bitlen,
                     traversed_hashmap_ptr,
                     traversed_hash_bitlen_u32,  // Cast int64_t to uint32_t
                     itopk_size_u32,             // Cast size_t to uint32_t
                     min_iterations_u32,         // Cast size_t to uint32_t
                     max_iterations_u32,         // Cast size_t to uint32_t
                     num_executed_iterations,
                     query_id_offset,  // Offset to add to query_id when calling filter
                     bitset_ptr,
                     bitset_len,
                     original_nbits);

  // Check for launch errors immediately
  cudaError_t launch_err = cudaPeekAtLastError();
  if (launch_err != cudaSuccess) {
    RAFT_LOG_ERROR("[JIT LAUNCHER] MULTI_CTA kernel launch error detected: %s (error code: %d)",
                   cudaGetErrorString(launch_err),
                   launch_err);
    RAFT_CUDA_TRY(launch_err);
  }

  // Synchronize to catch kernel execution errors before they propagate
  // This ensures the kernel completes before we return, preventing parameter lifetime issues
  cudaError_t sync_err = cudaStreamSynchronize(stream);
  if (sync_err != cudaSuccess) {
    RAFT_LOG_ERROR("[JIT LAUNCHER] MULTI_CTA kernel execution failed: %s (error code: %d)",
                   cudaGetErrorString(sync_err),
                   sync_err);
    RAFT_LOG_ERROR(
      "[JIT LAUNCHER] MULTI_CTA parameters: graph_degree=%u, itopk_size=%u, num_queries=%u, "
      "topk=%u, num_cta_per_query=%u, max_elements=%u",
      graph_degree_u32,
      itopk_size_u32,
      num_queries,
      topk,
      num_cta_per_query,
      max_elements);
    RAFT_LOG_ERROR(
      "[JIT LAUNCHER] MULTI_CTA pointers: topk_indices=%p, topk_distances=%p, graph=%p, "
      "dev_desc=%p",
      topk_indices_ptr,
      topk_distances_ptr,
      graph.data_handle(),
      dev_desc);
    RAFT_CUDA_TRY(sync_err);
  }

  RAFT_LOG_INFO("[JIT LAUNCHER] MULTI_CTA kernel completed successfully");
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
