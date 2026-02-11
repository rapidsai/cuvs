/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "search_single_cta_kernel_jit.cuh"

#include "bitonic.hpp"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_by_radix.cuh"
#include "topk_for_cagra/topk.h"
#include "utils.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/neighbors/common.hpp>

#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuda/atomic>
#include <cuda/std/atomic>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// JIT version of compute_distance_to_random_nodes - uses extern compute_distance
template <DescriptorType DescType,
          typename IndexT,
          typename DistanceT,
          typename DataT,
          cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS   = 0,
          uint32_t PQ_LEN    = 0,
          typename CodebookT = void>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_nodes_jit(
  IndexT* __restrict__ result_indices_ptr,
  DistanceT* __restrict__ result_distances_ptr,
  const DataT* dataset_ptr,
  const uint8_t* encoded_dataset_ptr,
  uint32_t smem_ws_ptr,
  IndexT dataset_size,
  uint32_t dim,
  uint32_t encoded_dataset_dim,
  uint32_t ld,
  uint32_t team_size_bitshift,
  const DistanceT* dataset_norms,
  const CodebookT* vq_code_book_ptr,
  const CodebookT* pq_code_book_ptr,
  const uint32_t num_pickup,
  const uint32_t num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* __restrict__ seed_ptr,
  const uint32_t num_seeds,
  IndexT* __restrict__ visited_hash_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hash_ptr,
  const uint32_t traversed_hash_bitlen,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1)
{
  constexpr unsigned warp_size = 32;
  const auto max_i = raft::round_up_safe<uint32_t>(num_pickup, warp_size >> team_size_bitshift);

  for (uint32_t i = threadIdx.x >> team_size_bitshift; i < max_i;
       i += (blockDim.x >> team_size_bitshift)) {
    const bool valid_i = (i < num_pickup);

    IndexT best_index_team_local    = raft::upper_bound<IndexT>();
    DistanceT best_norm2_team_local = raft::upper_bound<DistanceT>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      IndexT seed_index = 0;
      if (valid_i) {
        uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % dataset_size;
        }
      }

      DistanceT norm2 = 0;
      if constexpr (DescType == DescriptorType::Standard) {
        norm2 =
          valid_i
            ? compute_distance_standard<Metric,
                                        TeamSize,
                                        DatasetBlockDim,
                                        DataT,
                                        IndexT,
                                        DistanceT>(
                dataset_ptr, smem_ws_ptr, seed_index, dim, ld, team_size_bitshift, dataset_norms)
            : 0;
      } else if constexpr (DescType == DescriptorType::VPQ) {
        norm2 = valid_i ? compute_distance_vpq<Metric,
                                               TeamSize,
                                               DatasetBlockDim,
                                               PQ_BITS,
                                               PQ_LEN,
                                               CodebookT,
                                               DataT,
                                               IndexT,
                                               DistanceT>(encoded_dataset_ptr,
                                                          smem_ws_ptr,
                                                          seed_index,
                                                          encoded_dataset_dim,
                                                          vq_code_book_ptr,
                                                          pq_code_book_ptr,
                                                          team_size_bitshift)
                        : 0;
      }
      const auto norm2_sum = device::team_sum(norm2, team_size_bitshift);

      if (valid_i && (norm2_sum < best_norm2_team_local)) {
        best_norm2_team_local = norm2_sum;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x & ((1u << team_size_bitshift) - 1u);
    if (valid_i && lane_id == 0) {
      if (best_index_team_local != raft::upper_bound<IndexT>()) {
        if (hashmap::insert(visited_hash_ptr, visited_hash_bitlen, best_index_team_local) == 0) {
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        } else if ((traversed_hash_ptr != nullptr) &&
                   hashmap::search<IndexT, 1>(
                     traversed_hash_ptr, traversed_hash_bitlen, best_index_team_local)) {
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        }
      }
      result_distances_ptr[i] = best_norm2_team_local;
      result_indices_ptr[i]   = best_index_team_local;
    }
  }
}

// JIT version of compute_distance_to_child_nodes - uses extern compute_distance
template <DescriptorType DescType,
          typename IndexT,
          typename DistanceT,
          typename DataT,
          cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS           = 0,
          uint32_t PQ_LEN            = 0,
          typename CodebookT         = void,
          int STATIC_RESULT_POSITION = 1>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_child_nodes_jit(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  const DataT* dataset_ptr,
  const uint8_t* encoded_dataset_ptr,
  uint32_t smem_ws_ptr,
  uint32_t dim,
  uint32_t encoded_dataset_dim,
  uint32_t ld,
  uint32_t team_size_bitshift,
  const DistanceT* dataset_norms,
  const CodebookT* vq_code_book_ptr,
  const CodebookT* pq_code_book_ptr,
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  IndexT* __restrict__ visited_hashmap_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hashmap_ptr,
  const uint32_t traversed_hash_bitlen,
  const IndexT* __restrict__ parent_indices,
  const IndexT* __restrict__ internal_topk_list,
  const uint32_t search_width,
  int* __restrict__ result_position = nullptr,
  const int max_result_position     = 0)
{
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  constexpr IndexT invalid_index    = ~static_cast<IndexT>(0);

  // Read child indices of parents from knn graph and check if the distance computation is
  // necessary.
  for (uint32_t i = threadIdx.x; i < knn_k * search_width; i += blockDim.x) {
    const IndexT smem_parent_id = parent_indices[i / knn_k];
    IndexT child_id             = invalid_index;
    if (smem_parent_id != invalid_index) {
      const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
      child_id             = knn_graph[(i % knn_k) + (static_cast<int64_t>(knn_k) * parent_id)];
    }
    if (child_id != invalid_index) {
      if (hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child_id) == 0) {
        child_id = invalid_index;
      } else if ((traversed_hashmap_ptr != nullptr) &&
                 hashmap::search<IndexT, 1>(
                   traversed_hashmap_ptr, traversed_hash_bitlen, child_id)) {
        child_id = invalid_index;
      }
    }
    if (STATIC_RESULT_POSITION) {
      result_child_indices_ptr[i] = child_id;
    } else if (child_id != invalid_index) {
      int j                       = atomicSub(result_position, 1) - 1;
      result_child_indices_ptr[j] = child_id;
    }
  }
  __syncthreads();

  // Compute the distance to child nodes using extern compute_distance
  constexpr unsigned warp_size = 32;
  const auto num_k             = knn_k * search_width;
  const auto max_i     = raft::round_up_safe<uint32_t>(num_k, warp_size >> team_size_bitshift);
  const bool lead_lane = (threadIdx.x & ((1u << team_size_bitshift) - 1u)) == 0;
  const uint32_t ofst  = STATIC_RESULT_POSITION ? 0 : result_position[0];
  for (uint32_t i = threadIdx.x >> team_size_bitshift; i < max_i;
       i += blockDim.x >> team_size_bitshift) {
    const auto j        = i + ofst;
    const bool valid_i  = STATIC_RESULT_POSITION ? (j < num_k) : (j < max_result_position);
    const auto child_id = valid_i ? result_child_indices_ptr[j] : invalid_index;

    DistanceT child_dist = 0;
    if constexpr (DescType == DescriptorType::Standard) {
      child_dist = device::team_sum(
        (child_id != invalid_index)
          ? compute_distance_standard<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>(
              dataset_ptr, smem_ws_ptr, child_id, dim, ld, team_size_bitshift, dataset_norms)
          : (lead_lane ? raft::upper_bound<DistanceT>() : 0),
        team_size_bitshift);
    } else if constexpr (DescType == DescriptorType::VPQ) {
      child_dist = device::team_sum((child_id != invalid_index)
                                      ? compute_distance_vpq<Metric,
                                                             TeamSize,
                                                             DatasetBlockDim,
                                                             PQ_BITS,
                                                             PQ_LEN,
                                                             CodebookT,
                                                             DataT,
                                                             IndexT,
                                                             DistanceT>(encoded_dataset_ptr,
                                                                        smem_ws_ptr,
                                                                        child_id,
                                                                        encoded_dataset_dim,
                                                                        vq_code_book_ptr,
                                                                        pq_code_book_ptr,
                                                                        team_size_bitshift)
                                      : (lead_lane ? raft::upper_bound<DistanceT>() : 0),
                                    team_size_bitshift);
    }
    __syncwarp();

    // Store the distance
    if (valid_i && lead_lane) { result_child_distances_ptr[j] = child_dist; }
  }
}

// JIT version of search_core - uses extern functions instead of templated descriptor
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          DescriptorType DescType,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS   = 0,
          uint32_t PQ_LEN    = 0,
          typename CodebookT = void>
RAFT_DEVICE_INLINE_FUNCTION void search_core(uintptr_t result_indices_ptr,
                                             DistanceT* const result_distances_ptr,
                                             const std::uint32_t top_k,
                                             const DataT* const queries_ptr,
                                             const IndexT* const knn_graph,
                                             const std::uint32_t graph_degree,
                                             const SourceIndexT* source_indices_ptr,
                                             const unsigned num_distilation,
                                             const uint64_t rand_xor_mask,
                                             const IndexT* seed_ptr,
                                             const uint32_t num_seeds,
                                             IndexT* const visited_hashmap_ptr,
                                             const std::uint32_t max_candidates,
                                             const std::uint32_t max_itopk,
                                             const std::uint32_t internal_topk,
                                             const std::uint32_t search_width,
                                             const std::uint32_t min_iteration,
                                             const std::uint32_t max_iteration,
                                             std::uint32_t* const num_executed_iterations,
                                             const std::uint32_t hash_bitlen,
                                             const std::uint32_t small_hash_bitlen,
                                             const std::uint32_t small_hash_reset_interval,
                                             const std::uint32_t query_id,
                                             const DataT* dataset_ptr,
                                             const uint8_t* encoded_dataset_ptr,
                                             IndexT dataset_size,
                                             uint32_t dim,
                                             uint32_t encoded_dataset_dim,
                                             uint32_t ld,
                                             const DistanceT* dataset_norms,
                                             const CodebookT* vq_code_book_ptr,
                                             const CodebookT* pq_code_book_ptr)
{
  using LOAD_T = device::LOAD_128BIT_T;

  auto to_source_index = [source_indices_ptr](IndexT x) {
    return source_indices_ptr == nullptr ? static_cast<SourceIndexT>(x) : source_indices_ptr[x];
  };

#ifdef _CLK_BREAKDOWN
  std::uint64_t clk_init                 = 0;
  std::uint64_t clk_compute_1st_distance = 0;
  std::uint64_t clk_topk                 = 0;
  std::uint64_t clk_reset_hash           = 0;
  std::uint64_t clk_pickup_parents       = 0;
  std::uint64_t clk_restore_hash         = 0;
  std::uint64_t clk_compute_distance     = 0;
  std::uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  const auto result_buffer_size    = internal_topk + (search_width * graph_degree);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  const auto small_hash_size       = hashmap::get_size(small_hash_bitlen);

  // Compute smem_ws_size_in_bytes based on descriptor type
  uint32_t smem_ws_size_in_bytes = 0;
  if constexpr (DescType == DescriptorType::Standard) {
    using desc_type = cuvs::neighbors::cagra::detail::
      standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
    using QUERY_T = typename desc_type::QUERY_T;
    smem_ws_size_in_bytes =
      sizeof(desc_type) + raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  } else if constexpr (DescType == DescriptorType::VPQ) {
    using desc_type = cuvs::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<Metric,
                                                                                   TeamSize,
                                                                                   DatasetBlockDim,
                                                                                   PQ_BITS,
                                                                                   PQ_LEN,
                                                                                   CodebookT,
                                                                                   DataT,
                                                                                   IndexT,
                                                                                   DistanceT>;
    using QUERY_T   = typename desc_type::QUERY_T;
    constexpr uint32_t kSMemCodeBookSizeInBytes = (1 << PQ_BITS) * PQ_LEN * sizeof(CodebookT);
    smem_ws_size_in_bytes                       = sizeof(desc_type) + kSMemCodeBookSizeInBytes +
                            raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }

  // Set smem working buffer for the distance calculation using extern function
  uint32_t smem_ws_ptr = 0;
  if constexpr (DescType == DescriptorType::Standard) {
    smem_ws_ptr =
      setup_workspace_standard<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>(
        smem, queries_ptr, query_id, dataset_ptr, dataset_size, dim, ld, dataset_norms);
  } else if constexpr (DescType == DescriptorType::VPQ) {
    smem_ws_ptr = setup_workspace_vpq<Metric,
                                      TeamSize,
                                      DatasetBlockDim,
                                      PQ_BITS,
                                      PQ_LEN,
                                      CodebookT,
                                      DataT,
                                      IndexT,
                                      DistanceT>(smem,
                                                 queries_ptr,
                                                 query_id,
                                                 encoded_dataset_ptr,
                                                 encoded_dataset_dim,
                                                 vq_code_book_ptr,
                                                 pq_code_book_ptr,
                                                 dataset_size,
                                                 dim);
  }

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<IndexT*>(smem + smem_ws_size_in_bytes);
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DistanceT*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ visited_hash_buffer =
    reinterpret_cast<IndexT*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_list_buffer =
    reinterpret_cast<IndexT*>(visited_hash_buffer + small_hash_size);
  auto* __restrict__ topk_ws = reinterpret_cast<std::uint32_t*>(parent_list_buffer + search_width);
  auto* terminate_flag       = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto* __restrict__ smem_work_ptr = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  // A flag for filtering.
  auto filter_flag = terminate_flag;

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  // Init hashmap
  IndexT* local_visited_hashmap_ptr;
  if (small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * blockIdx.y);
  }
  hashmap::init(local_visited_hashmap_ptr, hash_bitlen, 0);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes using JIT version
  _CLK_START();
  const IndexT* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  constexpr uint32_t team_size_bits  = raft::Pow2<TeamSize>::Log2;
  compute_distance_to_random_nodes_jit<DescType,
                                       IndexT,
                                       DistanceT,
                                       DataT,
                                       Metric,
                                       TeamSize,
                                       DatasetBlockDim,
                                       PQ_BITS,
                                       PQ_LEN,
                                       CodebookT>(result_indices_buffer,
                                                  result_distances_buffer,
                                                  dataset_ptr,
                                                  encoded_dataset_ptr,
                                                  smem_ws_ptr,
                                                  dataset_size,
                                                  dim,
                                                  encoded_dataset_dim,
                                                  ld,
                                                  team_size_bits,
                                                  dataset_norms,
                                                  vq_code_book_ptr,
                                                  pq_code_book_ptr,
                                                  result_buffer_size,
                                                  num_distilation,
                                                  rand_xor_mask,
                                                  local_seed_ptr,
                                                  num_seeds,
                                                  local_visited_hashmap_ptr,
                                                  hash_bitlen,
                                                  (IndexT*)nullptr,
                                                  0);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  std::uint32_t iter = 0;
  while (1) {
    // sort
    if constexpr (TOPK_BY_BITONIC_SORT) {
      assert(blockDim.x >= 64);
      const bool bitonic_sort_and_full_multi_warps = (max_candidates > 128) ? true : false;

      // reset small-hash table.
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        unsigned hash_start_tid;
        if (blockDim.x == 32) {
          hash_start_tid = 0;
        } else if (blockDim.x == 64) {
          if (bitonic_sort_and_full_multi_warps || BITONIC_SORT_AND_MERGE_MULTI_WARPS) {
            hash_start_tid = 0;
          } else {
            hash_start_tid = 32;
          }
        } else {
          if (bitonic_sort_and_full_multi_warps || BITONIC_SORT_AND_MERGE_MULTI_WARPS) {
            hash_start_tid = 64;
          } else {
            hash_start_tid = 32;
          }
        }
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen, hash_start_tid);
        _CLK_REC(clk_reset_hash);
      }

      // topk with bitonic sort
      _CLK_START();
      // For JIT version, we always check filter_flag at runtime since sample_filter is extern
      if (*filter_flag != 0) {
        // Move the filtered out index to the end of the itopk list
        for (unsigned i = 0; i < search_width; i++) {
          move_invalid_to_end_of_list(
            result_indices_buffer, result_distances_buffer, internal_topk);
        }
        if (threadIdx.x == 0) { *terminate_flag = 0; }
      }
      topk_by_bitonic_sort_and_merge<BITONIC_SORT_AND_MERGE_MULTI_WARPS>(
        result_distances_buffer,
        result_indices_buffer,
        max_itopk,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        max_candidates,
        search_width * graph_degree,
        topk_ws,
        (iter == 0));
      __syncthreads();
      _CLK_REC(clk_topk);
    } else {
      _CLK_START();
      // topk with radix block sort
      topk_by_radix_sort<IndexT>{}(max_itopk,
                                   internal_topk,
                                   result_buffer_size,
                                   reinterpret_cast<std::uint32_t*>(result_distances_buffer),
                                   result_indices_buffer,
                                   reinterpret_cast<std::uint32_t*>(result_distances_buffer),
                                   result_indices_buffer,
                                   nullptr,
                                   topk_ws,
                                   true,
                                   smem_work_ptr);
      _CLK_REC(clk_topk);

      // reset small-hash table
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen);
        _CLK_REC(clk_reset_hash);
      }
    }
    __syncthreads();

    if (iter + 1 == max_iteration) { break; }

    // pick up next parents
    if (threadIdx.x < 32) {
      _CLK_START();
      pickup_next_parents<TOPK_BY_BITONIC_SORT, IndexT>(
        terminate_flag, parent_list_buffer, result_indices_buffer, internal_topk, search_width);
      _CLK_REC(clk_pickup_parents);
    }

    // restore small-hash table by putting internal-topk indices in it
    if ((iter + 1) % small_hash_reset_interval == 0) {
      const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
      _CLK_START();
      hashmap_restore(
        local_visited_hashmap_ptr, hash_bitlen, result_indices_buffer, internal_topk, first_tid);
      _CLK_REC(clk_restore_hash);
    }
    __syncthreads();

    if (*terminate_flag && iter >= min_iteration) { break; }

    // compute the norms between child nodes and query node using JIT version
    _CLK_START();
    compute_distance_to_child_nodes_jit<DescType,
                                        IndexT,
                                        DistanceT,
                                        DataT,
                                        Metric,
                                        TeamSize,
                                        DatasetBlockDim,
                                        PQ_BITS,
                                        PQ_LEN,
                                        CodebookT>(result_indices_buffer + internal_topk,
                                                   result_distances_buffer + internal_topk,
                                                   dataset_ptr,
                                                   encoded_dataset_ptr,
                                                   smem_ws_ptr,
                                                   dim,
                                                   encoded_dataset_dim,
                                                   ld,
                                                   team_size_bits,
                                                   dataset_norms,
                                                   vq_code_book_ptr,
                                                   pq_code_book_ptr,
                                                   knn_graph,
                                                   graph_degree,
                                                   local_visited_hashmap_ptr,
                                                   hash_bitlen,
                                                   (IndexT*)nullptr,
                                                   0,
                                                   parent_list_buffer,
                                                   result_indices_buffer,
                                                   search_width);
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering - use extern sample_filter function
    if (threadIdx.x == 0) { *filter_flag = 0; }
    __syncthreads();

    constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
    const IndexT invalid_index        = utils::get_max_value<IndexT>();

    for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
      if (parent_list_buffer[p] != invalid_index) {
        const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
        if (!sample_filter<SourceIndexT>(query_id, to_source_index(parent_id))) {
          result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DistanceT>();
          result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
          *filter_flag                                   = 1;
        }
      }
    }
    __syncthreads();

    iter++;
  }

  // Post process for filtering - use extern sample_filter function
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  const IndexT invalid_index        = utils::get_max_value<IndexT>();

  for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree; i += blockDim.x) {
    const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
    if (node_id != (invalid_index & ~index_msb_1_mask) &&
        !sample_filter<SourceIndexT>(query_id, to_source_index(node_id))) {
      result_distances_buffer[i] = utils::get_max_value<DistanceT>();
      result_indices_buffer[i]   = invalid_index;
    }
  }

  __syncthreads();
  // Move invalid index items to the end of the buffer without sorting the entire buffer
  using scan_op_t    = cub::WarpScan<unsigned>;
  auto& temp_storage = *reinterpret_cast<typename scan_op_t::TempStorage*>(smem_work_ptr);

  constexpr std::uint32_t warp_size = 32;
  if (threadIdx.x < warp_size) {
    std::uint32_t num_found_valid = 0;
    for (std::uint32_t buffer_offset = 0; buffer_offset < internal_topk;
         buffer_offset += warp_size) {
      const auto src_position = buffer_offset + threadIdx.x;
      const std::uint32_t is_valid_index =
        (result_indices_buffer[src_position] & (~index_msb_1_mask)) == invalid_index ? 0 : 1;
      std::uint32_t new_position;
      scan_op_t(temp_storage).InclusiveSum(is_valid_index, new_position);
      if (is_valid_index) {
        const auto dst_position               = num_found_valid + (new_position - 1);
        result_indices_buffer[dst_position]   = result_indices_buffer[src_position];
        result_distances_buffer[dst_position] = result_distances_buffer[src_position];
      }

      num_found_valid += new_position;
      for (std::uint32_t offset = (warp_size >> 1); offset > 0; offset >>= 1) {
        const auto v = raft::shfl_xor(num_found_valid, offset);
        if ((threadIdx.x & offset) == 0) { num_found_valid = v; }
      }

      if (num_found_valid >= top_k) { break; }
    }

    if (num_found_valid < top_k) {
      for (std::uint32_t i = num_found_valid + threadIdx.x; i < internal_topk; i += warp_size) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DistanceT>();
      }
    }
  }

  // If the sufficient number of valid indexes are not in the internal topk, pick up from the
  // candidate list.
  if (top_k > internal_topk || result_indices_buffer[top_k - 1] == invalid_index) {
    __syncthreads();
    topk_by_bitonic_sort_and_merge<BITONIC_SORT_AND_MERGE_MULTI_WARPS>(
      result_distances_buffer,
      result_indices_buffer,
      max_itopk,
      internal_topk,
      result_distances_buffer + internal_topk,
      result_indices_buffer + internal_topk,
      max_candidates,
      search_width * graph_degree,
      topk_ws,
      (iter == 0));
  }
  __syncthreads();

  // NB: The indices pointer is tagged with its element size.
  const uint32_t index_element_tag = result_indices_ptr & 0x3;
  result_indices_ptr ^= index_element_tag;
  auto write_indices =
    index_element_tag == 3
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint64_t*>(ptr)[i] = static_cast<uint64_t>(x); }
    : index_element_tag == 2
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint32_t*>(ptr)[i] = static_cast<uint32_t>(x); }
    : index_element_tag == 1
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint16_t*>(ptr)[i] = static_cast<uint16_t>(x); }
      : [](uintptr_t ptr, uint32_t i, SourceIndexT x) {
          reinterpret_cast<uint8_t*>(ptr)[i] = static_cast<uint8_t>(x);
        };
  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if constexpr (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }

    auto internal_index =
      result_indices_buffer[ii] & ~index_msb_1_mask;  // clear most significant bit
    auto source_index = to_source_index(internal_index);
    write_indices(result_indices_ptr, j, source_index);
  }
  if (threadIdx.x == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }
#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) && ((query_id * 3) % gridDim.y < 3)) {
    printf(
      "%s:%d "
      "query, %d, thread, %d"
      ", init, %lu"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", reset_hash, %lu"
      ", pickup_parents, %lu"
      ", restore_hash, %lu"
      ", distance, %lu"
      "\n",
      __FILE__,
      __LINE__,
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_reset_hash,
      clk_pickup_parents,
      clk_restore_hash,
      clk_compute_distance);
  }
#endif
}

// JIT kernel wrapper - calls search_core
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          DescriptorType DescType,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS       = 0,
          uint32_t PQ_LEN        = 0,
          typename CodebookT     = void,
          typename SampleFilterT = void>
RAFT_KERNEL __launch_bounds__(1024, 1)
  search_kernel_jit(uintptr_t result_indices_ptr,
                    DistanceT* const result_distances_ptr,
                    const std::uint32_t top_k,
                    const DataT* const queries_ptr,
                    const IndexT* const knn_graph,
                    const std::uint32_t graph_degree,
                    const SourceIndexT* source_indices_ptr,
                    const unsigned num_distilation,
                    const uint64_t rand_xor_mask,
                    const IndexT* seed_ptr,
                    const uint32_t num_seeds,
                    IndexT* const visited_hashmap_ptr,
                    const std::uint32_t max_candidates,
                    const std::uint32_t max_itopk,
                    const std::uint32_t internal_topk,
                    const std::uint32_t search_width,
                    const std::uint32_t min_iteration,
                    const std::uint32_t max_iteration,
                    std::uint32_t* const num_executed_iterations,
                    const std::uint32_t hash_bitlen,
                    const std::uint32_t small_hash_bitlen,
                    const std::uint32_t small_hash_reset_interval,
                    const DataT* dataset_ptr,
                    const uint8_t* encoded_dataset_ptr,
                    IndexT dataset_size,
                    uint32_t dim,
                    uint32_t encoded_dataset_dim,
                    uint32_t ld,
                    const DistanceT* dataset_norms,
                    const CodebookT* vq_code_book_ptr,
                    const CodebookT* pq_code_book_ptr,
                    SampleFilterT sample_filter)
{
  const auto query_id = blockIdx.y;
  search_core<TOPK_BY_BITONIC_SORT,
              BITONIC_SORT_AND_MERGE_MULTI_WARPS,
              DescType,
              DataT,
              IndexT,
              DistanceT,
              SourceIndexT,
              Metric,
              TeamSize,
              DatasetBlockDim,
              PQ_BITS,
              PQ_LEN,
              CodebookT>(result_indices_ptr,
                         result_distances_ptr,
                         top_k,
                         queries_ptr,
                         knn_graph,
                         graph_degree,
                         source_indices_ptr,
                         num_distilation,
                         rand_xor_mask,
                         seed_ptr,
                         num_seeds,
                         visited_hashmap_ptr,
                         max_candidates,
                         max_itopk,
                         internal_topk,
                         search_width,
                         min_iteration,
                         max_iteration,
                         num_executed_iterations,
                         hash_bitlen,
                         small_hash_bitlen,
                         small_hash_reset_interval,
                         query_id,
                         dataset_ptr,
                         encoded_dataset_ptr,
                         dataset_size,
                         dim,
                         encoded_dataset_dim,
                         ld,
                         dataset_norms,
                         vq_code_book_ptr,
                         pq_code_book_ptr);
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
