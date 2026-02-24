/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance-ext.cuh"
#include "../device_common.hpp"
#include "../hashmap.hpp"
#include "../utils.hpp"

#include <cuvs/distance/distance.hpp>  // For DistanceType enum
#include <raft/core/operators.hpp>     // For raft::upper_bound

#include <cstdint>
#include <cuda_fp16.h>
#include <type_traits>  // For std::is_same_v, std::true_type, std::false_type

// Include extern function declarations before namespace so they're available to kernel definitions
#include "../../jit_lto_kernels/filter_data.h"
#include "extern_device_functions.cuh"

namespace cuvs::neighbors::cagra::detail::multi_kernel_search {

// Helper to check if DescriptorT has kPqBits (VPQ descriptor)
template <typename T>
struct has_kpq_bits {
  template <typename U>
  static auto test(int) -> decltype(U::kPqBits, std::true_type{});
  template <typename>
  static std::false_type test(...);
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T>
inline constexpr bool has_kpq_bits_v = has_kpq_bits<T>::value;

// JIT version of random_pickup_kernel - uses dataset_descriptor_base_t* pointer
// Unified template parameters: TeamSize, DatasetBlockDim, PQ_BITS, PQ_LEN, CodebookT, QueryT
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
RAFT_KERNEL random_pickup_kernel_jit(
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const DataT* const queries_ptr,  // [num_queries, dataset_dim]
  const std::size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  IndexT* const result_indices_ptr,       // [num_queries, ldr]
  DistanceT* const result_distances_ptr,  // [num_queries, ldr]
  const std::uint32_t ldr,                // (*) ldr >= num_pickup
  IndexT* const visited_hashmap_ptr,      // [num_queries, 1 << bitlen]
  const std::uint32_t hash_bitlen)
{
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  // Get team_size_bits directly from base descriptor
  uint32_t team_size_bits = dataset_desc->team_size_bitshift();

  const auto ldb               = hashmap::get_size(hash_bitlen);
  const auto global_team_index = (blockIdx.x * blockDim.x + threadIdx.x) >> team_size_bits;
  const uint32_t query_id      = blockIdx.y;
  if (global_team_index >= num_pickup) { return; }
  extern __shared__ uint8_t smem[];

  // Set smem working buffer using unified setup_workspace
  // setup_workspace copies the descriptor to shared memory and returns base pointer to smem
  // descriptor NOTE: setup_workspace must be called by ALL threads (it uses __syncthreads())
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc =
    setup_workspace<TeamSize,
                    DatasetBlockDim,
                    PQ_BITS,
                    PQ_LEN,
                    CodebookT,
                    DataT,
                    IndexT,
                    DistanceT,
                    QueryT>(dataset_desc, smem, queries_ptr, query_id);
  __syncthreads();

  // Load args once for better performance (avoid repeated loads in the loop)
  using args_t = typename cuvs::neighbors::cagra::detail::
    dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t;
  args_t args         = smem_desc->args.load();
  IndexT dataset_size = smem_desc->size;

  INDEX_T best_index_team_local;
  DISTANCE_T best_norm2_team_local = utils::get_max_value<DISTANCE_T>();
  for (unsigned i = 0; i < num_distilation; i++) {
    INDEX_T seed_index;
    if (seed_ptr && (global_team_index < num_seeds)) {
      seed_index = seed_ptr[global_team_index + (num_seeds * query_id)];
    } else {
      // Chose a seed node randomly
      seed_index = device::xorshift64((global_team_index ^ rand_xor_mask) * (i + 1)) % dataset_size;
    }

    // CRITICAL: ALL threads in the team must participate in compute_distance and team_sum
    // Otherwise warp shuffles will hang. Each thread calls the unified extern function to get
    // its per-thread distance, then team_sum reduces across all threads in the team.
    DistanceT per_thread_norm2 = 0;
    // Use unified compute_distance function (planner links standard or VPQ fragment at runtime)
    per_thread_norm2 = compute_distance<TeamSize,
                                        DatasetBlockDim,
                                        PQ_BITS,
                                        PQ_LEN,
                                        CodebookT,
                                        DataT,
                                        IndexT,
                                        DistanceT,
                                        QueryT>(args, seed_index);
    // Now ALL threads in the team participate in team_sum
    const auto norm2 = device::team_sum(per_thread_norm2, team_size_bits);

    if (norm2 < best_norm2_team_local) {
      best_norm2_team_local = norm2;
      best_index_team_local = seed_index;
    }
  }

  const auto store_gmem_index = global_team_index + (ldr * query_id);
  if ((threadIdx.x & ((1u << team_size_bits) - 1u)) == 0) {
    if (hashmap::insert(
          visited_hashmap_ptr + (ldb * query_id), hash_bitlen, best_index_team_local)) {
      result_distances_ptr[store_gmem_index] = best_norm2_team_local;
      result_indices_ptr[store_gmem_index]   = best_index_team_local;
    } else {
      result_distances_ptr[store_gmem_index] = utils::get_max_value<DISTANCE_T>();
      result_indices_ptr[store_gmem_index]   = utils::get_max_value<INDEX_T>();
    }
  }
}

// JIT version of compute_distance_to_child_nodes_kernel - uses extern functions with void*
// descriptor Unified template parameters: TeamSize, DatasetBlockDim, PQ_BITS, PQ_LEN, CodebookT,
// QueryT
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT,
          typename SourceIndexT,
          typename SAMPLE_FILTER_T>
RAFT_KERNEL compute_distance_to_child_nodes_kernel_jit(
  const IndexT* const parent_node_list,  // [num_queries, search_width]
  IndexT* const parent_candidates_ptr,   // [num_queries, search_width]
  DistanceT* const parent_distance_ptr,  // [num_queries, search_width]
  const std::size_t lds,
  const std::uint32_t search_width,
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const IndexT* const neighbor_graph_ptr,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const DataT* query_ptr,             // [num_queries, data_dim]
  IndexT* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t hash_bitlen,
  IndexT* const result_indices_ptr,       // [num_queries, ldd]
  DistanceT* const result_distances_ptr,  // [num_queries, ldd]
  const std::uint32_t ldd,                // (*) ldd >= search_width * graph_degree
  SAMPLE_FILTER_T sample_filter)
{
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  // Get team_size_bits directly from base descriptor
  uint32_t team_size_bits = dataset_desc->team_size_bitshift();

  const auto team_size      = 1u << team_size_bits;
  const uint32_t ldb        = hashmap::get_size(hash_bitlen);
  const auto tid            = threadIdx.x + blockDim.x * blockIdx.x;
  const auto global_team_id = tid >> team_size_bits;
  const auto query_id       = blockIdx.y;

  extern __shared__ uint8_t smem[];
  // Load a query using unified setup_workspace
  // setup_workspace copies the descriptor to shared memory and returns base pointer to smem
  // descriptor NOTE: setup_workspace must be called by ALL threads (it uses __syncthreads())
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc =
    setup_workspace<TeamSize,
                    DatasetBlockDim,
                    PQ_BITS,
                    PQ_LEN,
                    CodebookT,
                    DataT,
                    IndexT,
                    DistanceT,
                    QueryT>(dataset_desc, smem, query_ptr, query_id);

  __syncthreads();
  if (global_team_id >= search_width * graph_degree) { return; }

  const std::size_t parent_list_index =
    parent_node_list[global_team_id / graph_degree + (search_width * blockIdx.y)];

  if (parent_list_index == utils::get_max_value<INDEX_T>()) { return; }

  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const auto raw_parent_index        = parent_candidates_ptr[parent_list_index + (lds * query_id)];

  if (raw_parent_index == utils::get_max_value<INDEX_T>()) {
    result_distances_ptr[ldd * blockIdx.y + global_team_id] = utils::get_max_value<DISTANCE_T>();
    return;
  }
  const auto parent_index = raw_parent_index & ~index_msb_1_mask;

  const auto neighbor_list_head_ptr = neighbor_graph_ptr + (graph_degree * parent_index);

  const std::size_t child_id = neighbor_list_head_ptr[global_team_id % graph_degree];

  const auto compute_distance_flag = hashmap::insert<INDEX_T>(
    team_size, visited_hashmap_ptr + (ldb * blockIdx.y), hash_bitlen, child_id);

  // Load args once for better performance (avoid repeated loads)
  using args_t = typename cuvs::neighbors::cagra::detail::
    dataset_descriptor_base_t<DataT, INDEX_T, DISTANCE_T>::args_t;
  args_t args = smem_desc->args.load();

  // CRITICAL: ALL threads in the team must participate in compute_distance and team_sum
  // Otherwise warp shuffles will hang. Each thread calls the unified extern function to get
  // its per-thread distance, then team_sum reduces across all threads in the team.
  DISTANCE_T per_thread_norm2 = 0;
  if (compute_distance_flag) {
    // Use unified compute_distance function (planner links standard or VPQ fragment at runtime)
    per_thread_norm2 = compute_distance<TeamSize,
                                        DatasetBlockDim,
                                        PQ_BITS,
                                        PQ_LEN,
                                        CodebookT,
                                        DataT,
                                        IndexT,
                                        DistanceT,
                                        QueryT>(args, child_id);
  }
  // Now ALL threads in the team participate in team_sum
  DISTANCE_T norm2 = device::team_sum(per_thread_norm2, team_size_bits);

  if (compute_distance_flag) {
    if ((threadIdx.x & (team_size - 1)) == 0) {
      result_indices_ptr[ldd * blockIdx.y + global_team_id]   = child_id;
      result_distances_ptr[ldd * blockIdx.y + global_team_id] = norm2;
    }
  } else {
    if ((threadIdx.x & (team_size - 1)) == 0) {
      result_distances_ptr[ldd * blockIdx.y + global_team_id] = utils::get_max_value<DISTANCE_T>();
    }
  }

  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              cuvs::neighbors::filtering::none_sample_filter>::value) {
    if (!sample_filter(
          query_id,
          source_indices_ptr == nullptr ? parent_index : source_indices_ptr[parent_index])) {
      parent_candidates_ptr[parent_list_index + (lds * query_id)] = utils::get_max_value<INDEX_T>();
      parent_distance_ptr[parent_list_index + (lds * query_id)] =
        utils::get_max_value<DISTANCE_T>();
    }
  }
}

// JIT version of apply_filter_kernel - uses extern sample_filter function
// Bitset data is passed as kernel parameters (matching non-JIT where filter object contains
// bitset_view) The bitset data is in global memory (not shared memory), just like non-JIT
using cuvs::neighbors::detail::sample_filter;
template <typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
RAFT_KERNEL apply_filter_kernel_jit(
  const SourceIndexT* source_indices_ptr,  // [num_queries, search_width]
  IndexT* const result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::size_t lds,
  const std::uint32_t result_buffer_size,
  const std::uint32_t num_queries,
  const IndexT query_id_offset,
  uint32_t* bitset_ptr,         // Bitset data pointer (nullptr for none_filter) - in global memory
  SourceIndexT bitset_len,      // Bitset length
  SourceIndexT original_nbits)  // Original number of bits
{
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  const auto tid                    = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= result_buffer_size * num_queries) { return; }
  const auto i     = tid % result_buffer_size;
  const auto j     = tid / result_buffer_size;
  const auto index = i + j * lds;

  if (result_indices_ptr[index] != ~index_msb_1_mask) {
    // Use extern sample_filter function with 3 params: query_id, node_id, filter_data
    // filter_data is a void* pointer to bitset_filter_data_t (or nullptr for none_filter)
    SourceIndexT node_id = source_indices_ptr == nullptr
                             ? static_cast<SourceIndexT>(result_indices_ptr[index])
                             : source_indices_ptr[result_indices_ptr[index]];

    // Construct filter_data struct in registers (bitset data is in global memory)
    cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT> filter_data(
      bitset_ptr, bitset_len, original_nbits);

    if (!sample_filter<SourceIndexT>(
          query_id_offset + j, node_id, bitset_ptr != nullptr ? &filter_data : nullptr)) {
      result_indices_ptr[index]   = utils::get_max_value<IndexT>();
      result_distances_ptr[index] = utils::get_max_value<DistanceT>();
    }
  }
}

}  // namespace cuvs::neighbors::cagra::detail::multi_kernel_search
