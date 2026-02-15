/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance-ext.cuh"
#include "../device_common.hpp"
#include "../hashmap.hpp"
#include "../utils.hpp"
// Note:
// - Extern function declarations (setup_workspace_standard, compute_distance_standard, etc.) use
// types from compute_distance-ext.cuh
// - Type definitions (standard_dataset_descriptor_t, etc.) are in the -impl.cuh files, included by
// the .cu.in files for template instantiation
// - pickup_next_parent and topk_by_bitonic_sort_wrapper_* are included via
// search_multi_cta_helpers.cuh in the .cu.in file

#include <cuvs/distance/distance.hpp>   // For DistanceType enum
#include <raft/core/operators.hpp>      // For raft::upper_bound
#include <raft/util/integer_utils.hpp>  // For raft::round_up_safe

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>  // For std::is_same_v, std::true_type, std::false_type

#ifdef _CLK_BREAKDOWN
#include <cstdio>  // For printf in debug code
#endif

// Include extern function declarations before namespace so they're available to kernel definitions
#include "extern_device_functions.cuh"
#include "filter_data.h"
// Include shared JIT device functions before namespace so they're available to kernel definitions
#include "device_common_jit.cuh"

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

// Helper to check if DescriptorT has kPqBits (VPQ descriptor) - use shared version
// Use fully qualified name since it's a template variable
using cuvs::neighbors::cagra::detail::device::has_kpq_bits_v;

// sample_filter is declared in extern_device_functions.cuh

// JIT versions of compute_distance_to_random_nodes and compute_distance_to_child_nodes
// are now shared in device_common_jit.cuh - use fully qualified names
using cuvs::neighbors::cagra::detail::device::compute_distance_to_child_nodes_jit;
using cuvs::neighbors::cagra::detail::device::compute_distance_to_random_nodes_jit;

// JIT version of search_kernel - uses extern functions with concrete descriptor type
// Filter is linked separately via JIT LTO, so we use none_sample_filter directly
template <typename DescriptorT,  // Concrete descriptor type
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__global__ __launch_bounds__(1024, 1) void search_kernel_jit(
  IndexT* const result_indices_ptr,       // [num_queries, num_cta_per_query, itopk_size]
  DistanceT* const result_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  const DescriptorT* dataset_desc,        // Concrete descriptor type from template
  const DataT* const queries_ptr,         // [num_queries, dataset_dim]
  const IndexT* const knn_graph,          // [dataset_size, graph_degree]
  const uint32_t max_elements,
  const uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,  // [num_queries, search_width]
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  const uint32_t visited_hash_bitlen,
  IndexT* const traversed_hashmap_ptr,  // [num_queries, 1 << traversed_hash_bitlen]
  const uint32_t traversed_hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations, /* stats */
  uint32_t* bitset_ptr,                    // Bitset data pointer (nullptr for none_filter)
  SourceIndexT bitset_len,                 // Bitset length
  SourceIndexT original_nbits)             // Original number of bits
{
  printf("IN THE KERNEL\n");
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  auto to_source_index = [source_indices_ptr](INDEX_T x) {
    return source_indices_ptr == nullptr ? static_cast<SourceIndexT>(x) : source_indices_ptr[x];
  };

  // CRITICAL DEBUG: Write to result buffer IMMEDIATELY to verify kernel is executing
  // Write a magic value that we can check on host - do this before ANY other code
  // Write from the first thread of the first block to maximize chance of execution
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && result_distances_ptr != nullptr &&
      result_indices_ptr != nullptr) {
    // Write magic value to first distance to verify kernel execution
    if (result_distances_ptr != nullptr) {
      *result_distances_ptr = static_cast<DistanceT>(3735928559.0f);  // 0xDEADBEEF as float
    }
    // Also write to indices
    if (result_indices_ptr != nullptr) { *result_indices_ptr = static_cast<IndexT>(0xCAFEBABE); }

    // Debug: Check if descriptor runtime values match kernel compile-time constants
    // The kernel uses DescriptorT::kTeamSize and DescriptorT::kDatasetBlockDim (compile-time)
    // The descriptor object has runtime values that should match
    uint32_t desc_team_size_bitshift  = dataset_desc->team_size_bitshift();
    uint32_t desc_team_size_actual    = 1u << desc_team_size_bitshift;
    uint32_t kernel_team_size         = DescriptorT::kTeamSize;
    uint32_t kernel_dataset_block_dim = DescriptorT::kDatasetBlockDim;

    // For standard descriptors, dataset_block_dim is stored in args.extra_word1 as 'ld'
    // For VPQ descriptors, it's a compile-time constant only
    uint32_t desc_dataset_block_dim = kernel_dataset_block_dim;  // Use compile-time constant
    if constexpr (!has_kpq_bits_v<DescriptorT>) {
      // Standard descriptor - can read from args.ld
      desc_dataset_block_dim = DescriptorT::ld(dataset_desc->args);
    }

    printf("JIT KERNEL EXECUTING: threadIdx=0, wrote magic values\n");
    printf("JIT KERNEL: Descriptor team_size (from bitshift): %u, Kernel kTeamSize: %u\n",
           desc_team_size_actual,
           kernel_team_size);
    printf("JIT KERNEL: Descriptor dataset_block_dim: %u, Kernel kDatasetBlockDim: %u\n",
           desc_dataset_block_dim,
           kernel_dataset_block_dim);
    if (desc_team_size_actual != kernel_team_size ||
        desc_dataset_block_dim != kernel_dataset_block_dim) {
      printf(
        "JIT KERNEL ERROR: Parameter mismatch! team_size: %u vs %u, dataset_block_dim: %u vs %u\n",
        desc_team_size_actual,
        kernel_team_size,
        desc_dataset_block_dim,
        kernel_dataset_block_dim);
    } else {
      printf("JIT KERNEL: Parameters match correctly\n");
    }
  }
  __syncthreads();

  const auto num_queries       = gridDim.y;
  const auto query_id          = blockIdx.y;
  const auto num_cta_per_query = gridDim.x;
  const auto cta_id            = blockIdx.x;  // local CTA ID

#ifdef _CLK_BREAKDOWN
  uint64_t clk_init                 = 0;
  uint64_t clk_compute_1st_distance = 0;
  uint64_t clk_topk                 = 0;
  uint64_t clk_pickup_parents       = 0;
  uint64_t clk_compute_distance     = 0;
  uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  // +----------------+---------+---------------------------+
  // | internal_top_k | padding | neighbors of parent nodes |
  // | <itopk_size>   | upto 32 | <graph_degree>            |
  // +----------------+---------+---------------------------+
  // |<---        result_buffer_size_32                 --->|
  const auto result_buffer_size    = itopk_size + graph_degree;
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  assert(result_buffer_size_32 <= max_elements);

  // Get smem_ws_size_in_bytes using static method (dim is in descriptor args)
  uint32_t dim                   = dataset_desc->args.dim;
  uint32_t smem_ws_size_in_bytes = DescriptorT::get_smem_ws_size_in_bytes(dim);

  // Set smem working buffer for the distance calculation using extern function
  // setup_workspace copies the descriptor to shared memory and returns pointer to smem descriptor
  // NOTE: setup_workspace must be called by ALL threads (it uses __syncthreads())
  const DescriptorT* smem_desc = nullptr;
  // Check if DescriptorT is a standard_dataset_descriptor_t by checking if it doesn't have kPqBits
  // (standard descriptors don't have kPqBits, VPQ descriptors do)
  if constexpr (!has_kpq_bits_v<DescriptorT>) {
    // Standard descriptor - use the metric from the descriptor type itself
    smem_desc = setup_workspace_standard<DescriptorT::kMetric,
                                         DescriptorT::kTeamSize,
                                         DescriptorT::kDatasetBlockDim,
                                         DataT,
                                         IndexT,
                                         DistanceT>(dataset_desc, smem, queries_ptr, query_id);
  } else {
    // Must be cagra_q_dataset_descriptor_t - use the metric from the descriptor type itself
    smem_desc = setup_workspace_vpq<DescriptorT::kMetric,
                                    DescriptorT::kTeamSize,
                                    DescriptorT::kDatasetBlockDim,
                                    DescriptorT::kPqBits,
                                    DescriptorT::kPqLen,
                                    typename DescriptorT::CODE_BOOK_T,
                                    DataT,
                                    IndexT,
                                    DistanceT>(dataset_desc, smem, queries_ptr, query_id);
  }

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + smem_ws_size_in_bytes);
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ local_visited_hashmap_ptr =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_indices_buffer =
    reinterpret_cast<INDEX_T*>(local_visited_hashmap_ptr + hashmap::get_size(visited_hash_bitlen));
  auto* __restrict__ result_position = reinterpret_cast<int*>(parent_indices_buffer + 1);

  INDEX_T* const local_traversed_hashmap_ptr =
    traversed_hashmap_ptr + (hashmap::get_size(traversed_hash_bitlen) * query_id);

  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
    result_indices_buffer[i]   = invalid_index;
    result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
  }
  hashmap::init<INDEX_T>(local_visited_hashmap_ptr, visited_hash_bitlen);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes using JIT version
  _CLK_START();
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  uint32_t block_id                   = cta_id + (num_cta_per_query * query_id);
  uint32_t num_blocks                 = num_cta_per_query * num_queries;

  compute_distance_to_random_nodes_jit<DescriptorT, IndexT, DistanceT, DataT>(
    result_indices_buffer,
    result_distances_buffer,
    smem_desc,
    graph_degree,
    num_distilation,
    rand_xor_mask,
    local_seed_ptr,
    num_seeds,
    local_visited_hashmap_ptr,
    visited_hash_bitlen,
    local_traversed_hashmap_ptr,
    traversed_hash_bitlen,
    block_id,
    num_blocks);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  uint32_t iter = 0;
  while (1) {
    _CLK_START();
    if (threadIdx.x < 32) {
      // [1st warp] Topk with bitonic sort
      if constexpr (std::is_same_v<INDEX_T, uint32_t>) {
        // use a non-template wrapper function to avoid pre-inlining the topk_by_bitonic_sort
        // function (vs post-inlining, this impacts register pressure)
        if (max_elements <= 64) {
          topk_by_bitonic_sort_wrapper_64(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else if (max_elements <= 128) {
          topk_by_bitonic_sort_wrapper_128(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else {
          assert(max_elements <= 256);
          topk_by_bitonic_sort_wrapper_256(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        }
      } else {
        if (max_elements <= 64) {
          topk_by_bitonic_sort<64, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else if (max_elements <= 128) {
          topk_by_bitonic_sort<128, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        } else {
          assert(max_elements <= 256);
          topk_by_bitonic_sort<256, INDEX_T>(
            result_distances_buffer, result_indices_buffer, result_buffer_size_32);
        }
      }
    }
    __syncthreads();
    _CLK_REC(clk_topk);

    if (iter + 1 >= max_iteration) { break; }

    _CLK_START();
    if (threadIdx.x < 32) {
      // [1st warp] Pick up a next parent
      pickup_next_parent<INDEX_T, DISTANCE_T>(parent_indices_buffer,
                                              result_indices_buffer,
                                              result_distances_buffer,
                                              local_traversed_hashmap_ptr,
                                              traversed_hash_bitlen);
    } else {
      // [Other warps] Reset visited hashmap
      hashmap::init<INDEX_T>(local_visited_hashmap_ptr, visited_hash_bitlen, 32);
    }
    __syncthreads();
    _CLK_REC(clk_pickup_parents);

    if ((parent_indices_buffer[0] == invalid_index) && (iter >= min_iteration)) { break; }

    _CLK_START();
    for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index) { continue; }
      if ((i >= itopk_size) && (index & index_msb_1_mask)) {
        // Remove nodes kicked out of the itopk list from the traversed hash table.
        hashmap::remove<INDEX_T>(
          local_traversed_hashmap_ptr, traversed_hash_bitlen, index & ~index_msb_1_mask);
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      } else {
        // Restore visited hashmap by putting nodes on result buffer in it.
        index &= ~index_msb_1_mask;
        hashmap::insert(local_visited_hashmap_ptr, visited_hash_bitlen, index);
      }
    }
    // Initialize buffer for compute_distance_to_child_nodes.
    if (threadIdx.x == blockDim.x - 1) { result_position[0] = result_buffer_size_32; }
    __syncthreads();

    // Compute the norms between child nodes and query node using JIT version
    compute_distance_to_child_nodes_jit<DescriptorT, IndexT, DistanceT, DataT, 0>(
      result_indices_buffer,
      result_distances_buffer,
      smem_desc,
      knn_graph,
      graph_degree,
      local_visited_hashmap_ptr,
      visited_hash_bitlen,
      local_traversed_hashmap_ptr,
      traversed_hash_bitlen,
      parent_indices_buffer,
      result_indices_buffer,
      1,
      result_position,
      result_buffer_size_32);
    __syncthreads();

    // Check the state of the nodes in the result buffer which were not updated
    // by the compute_distance_to_child_nodes above, and if it cannot be used as
    // a parent node, it is deactivated.
    for (uint32_t i = threadIdx.x; i < result_position[0]; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index || index & index_msb_1_mask) { continue; }
      if (hashmap::search<INDEX_T, 1>(local_traversed_hashmap_ptr, traversed_hash_bitlen, index)) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering - use extern sample_filter function (linked via JIT LTO)
    for (unsigned p = threadIdx.x; p < 1; p += blockDim.x) {
      if (parent_indices_buffer[p] != invalid_index) {
        const auto parent_id = result_indices_buffer[parent_indices_buffer[p]] & ~index_msb_1_mask;
        // Construct filter_data struct (bitset data is in global memory)
        cuvs::neighbors::cagra::detail::bitset_filter_data_t<SourceIndexT> filter_data(
          bitset_ptr, bitset_len, original_nbits);
        if (!sample_filter<SourceIndexT>(query_id,
                                         to_source_index(parent_id),
                                         bitset_ptr != nullptr ? &filter_data : nullptr)) {
          // If the parent must not be in the resulting top-k list, remove from the parent list
          result_distances_buffer[parent_indices_buffer[p]] = utils::get_max_value<DISTANCE_T>();
          result_indices_buffer[parent_indices_buffer[p]]   = invalid_index;
        }
      }
    }
    __syncthreads();

    iter++;
  }

  // Filtering - use extern sample_filter function (linked via JIT LTO)
  for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
    INDEX_T index = result_indices_buffer[i];
    if (index == invalid_index) { continue; }
    index &= ~index_msb_1_mask;
    // Construct filter_data struct (bitset data is in global memory)
    cuvs::neighbors::cagra::detail::bitset_filter_data_t<SourceIndexT> filter_data(
      bitset_ptr, bitset_len, original_nbits);
    if (!sample_filter<SourceIndexT>(
          query_id, to_source_index(index), bitset_ptr != nullptr ? &filter_data : nullptr)) {
      result_indices_buffer[i]   = invalid_index;
      result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
    }
  }
  __syncthreads();

  // Output search results (1st warp only).
  if (threadIdx.x < 32) {
    // Debug: print buffer contents before output
    if (query_id == 0 && cta_id == 0 && threadIdx.x < 5) {
      printf("JIT pre-output: i=%u idx=%u dist=%.6f\n",
             threadIdx.x,
             result_indices_buffer[threadIdx.x],
             (float)result_distances_buffer[threadIdx.x]);
    }
    uint32_t offset = 0;
    for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += 32) {
      INDEX_T index = result_indices_buffer[i];
      bool is_valid = false;
      if (index != invalid_index) {
        if (index & index_msb_1_mask) {
          is_valid = true;
          index &= ~index_msb_1_mask;
        } else if ((offset < itopk_size) &&
                   hashmap::insert<INDEX_T, 1>(
                     local_traversed_hashmap_ptr, traversed_hash_bitlen, index)) {
          // If a node that is not used as a parent can be inserted into
          // the traversed hash table, it is considered a valid result.
          is_valid = true;
        }
      }
      const auto mask = __ballot_sync(0xffffffff, is_valid);
      if (is_valid) {
        const auto j = offset + __popc(mask & ((1 << threadIdx.x) - 1));
        if (j < itopk_size) {
          uint32_t k            = j + (itopk_size * (cta_id + (num_cta_per_query * query_id)));
          result_indices_ptr[k] = index & ~index_msb_1_mask;
          if (result_distances_ptr != nullptr) {
            DISTANCE_T dist         = result_distances_buffer[i];
            result_distances_ptr[k] = dist;
            // Debug: print first query, first CTA, first few results
            if (query_id == 0 && cta_id == 0 && j < 5) {
              printf("JIT: query=%u cta=%u j=%u i=%u idx=%u dist=%.6f buf_dist=%.6f\n",
                     query_id,
                     cta_id,
                     j,
                     i,
                     index & ~index_msb_1_mask,
                     (float)dist,
                     (float)result_distances_buffer[i]);
            }
          }
        } else {
          // If it is valid and registered in the traversed hash table but is
          // not output as a result, it is removed from the hash table.
          hashmap::remove<INDEX_T>(local_traversed_hashmap_ptr, traversed_hash_bitlen, index);
        }
      }
      offset += __popc(mask);
    }
    // If the number of outputs is insufficient, fill in with invalid results.
    for (uint32_t i = offset + threadIdx.x; i < itopk_size; i += 32) {
      uint32_t k            = i + (itopk_size * (cta_id + (num_cta_per_query * query_id)));
      result_indices_ptr[k] = invalid_index;
      if (result_distances_ptr != nullptr) {
        result_distances_ptr[k] = utils::get_max_value<DISTANCE_T>();
      }
    }
  }

  if (threadIdx.x == 0 && cta_id == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }

#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) && (blockIdx.x == 0) &&
      ((query_id * 3) % gridDim.y < 3)) {
    printf(
      "%s:%d "
      "query, %d, thread, %d"
      ", init, %lu"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", pickup_parents, %lu"
      ", distance, %lu"
      "\n",
      __FILE__,
      __LINE__,
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_pickup_parents,
      clk_compute_distance);
  }
#endif
}

}  // namespace cuvs::neighbors::cagra::detail::multi_cta_search
