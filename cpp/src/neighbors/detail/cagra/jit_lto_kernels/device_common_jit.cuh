/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../device_common.hpp"
#include "../hashmap.hpp"
#include "../utils.hpp"
#include "extern_device_functions.cuh"

#include <cuvs/distance/distance.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::cagra::detail {
namespace device {

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

// JIT version of compute_distance_to_random_nodes - uses dataset_descriptor_base_t* pointer
// Shared between single_cta and multi_cta JIT kernels
// Unified template parameters: TeamSize, DatasetBlockDim, PQ_BITS, PQ_LEN, CodebookT, QueryT
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename IndexT,
          typename DistanceT,
          typename DataT,
          typename QueryT>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_nodes_jit(
  IndexT* __restrict__ result_indices_ptr,       // [num_pickup]
  DistanceT* __restrict__ result_distances_ptr,  // [num_pickup]
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const uint32_t num_pickup,
  const uint32_t num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* __restrict__ seed_ptr,  // [num_seeds]
  const uint32_t num_seeds,
  IndexT* __restrict__ visited_hash_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hash_ptr,
  const uint32_t traversed_hash_bitlen,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1)
{
  constexpr unsigned warp_size = 32;

  // Get team_size_bits and args directly from base descriptor
  using args_t = typename cuvs::neighbors::cagra::detail::
    dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t;

  // Use team_size_bitshift_from_smem since smem_desc is in shared memory
  uint32_t team_size_bits = smem_desc->team_size_bitshift_from_smem();
  args_t args             = smem_desc->args.load();
  IndexT dataset_size     = smem_desc->size;

  const auto max_i = raft::round_up_safe<uint32_t>(num_pickup, warp_size >> team_size_bits);

  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += (blockDim.x >> team_size_bits)) {
    const bool valid_i = (i < num_pickup);

    IndexT best_index_team_local    = raft::upper_bound<IndexT>();
    DistanceT best_norm2_team_local = raft::upper_bound<DistanceT>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      // Select a node randomly and compute the distance to it
      IndexT seed_index = 0;
      if (valid_i) {
        uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % dataset_size;
        }
      }

      // CRITICAL: ALL threads in the team must participate in compute_distance and team_sum
      // Otherwise warp shuffles will hang. Each thread calls the unified extern function to get
      // its per-thread distance, then team_sum reduces across all threads in the team.
      DistanceT per_thread_norm2 = 0;
      if (valid_i) {
        // Use unified compute_distance function (links standard or VPQ fragment at runtime)
        per_thread_norm2 = compute_distance<TeamSize,
                                            DatasetBlockDim,
                                            PQ_BITS,
                                            PQ_LEN,
                                            CodebookT,
                                            DataT,
                                            IndexT,
                                            DistanceT,
                                            QueryT>(args, seed_index);
      }
      // Now ALL threads in the team participate in team_sum
      const auto norm2_sum = device::team_sum(per_thread_norm2, team_size_bits);

      if (valid_i && (norm2_sum < best_norm2_team_local)) {
        best_norm2_team_local = norm2_sum;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x & ((1u << team_size_bits) - 1u);
    if (valid_i && lane_id == 0) {
      if (best_index_team_local != raft::upper_bound<IndexT>()) {
        if (hashmap::insert(visited_hash_ptr, visited_hash_bitlen, best_index_team_local) == 0) {
          // Deactivate this entry as insertion into visited hash table has failed.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        } else if ((traversed_hash_ptr != nullptr) &&
                   hashmap::search<IndexT, 1>(
                     traversed_hash_ptr, traversed_hash_bitlen, best_index_team_local)) {
          // Deactivate this entry as it has been already used by others.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        }
      }
      result_distances_ptr[i] = best_norm2_team_local;
      result_indices_ptr[i]   = best_index_team_local;
    }
  }
}

// JIT version of compute_distance_to_child_nodes - uses dataset_descriptor_base_t* pointer
// Shared between single_cta and multi_cta JIT kernels
// Unified template parameters: TeamSize, DatasetBlockDim, PQ_BITS, PQ_LEN, CodebookT, QueryT
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename IndexT,
          typename DistanceT,
          typename DataT,
          typename QueryT,
          int STATIC_RESULT_POSITION = 1>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_child_nodes_jit(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
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

  // Compute the distance to child nodes using unified extern compute_distance
  constexpr unsigned warp_size = 32;

  // Get team_size_bits and args directly from base descriptor
  using args_t = typename cuvs::neighbors::cagra::detail::
    dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t;

  // Use team_size_bitshift_from_smem since smem_desc is in shared memory
  uint32_t team_size_bits = smem_desc->team_size_bitshift_from_smem();
  args_t args             = smem_desc->args.load();

  const auto num_k     = knn_k * search_width;
  const auto max_i     = raft::round_up_safe(num_k, warp_size >> team_size_bits);
  const bool lead_lane = (threadIdx.x & ((1u << team_size_bits) - 1u)) == 0;
  const uint32_t ofst  = STATIC_RESULT_POSITION ? 0 : result_position[0];

  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += blockDim.x >> team_size_bits) {
    const auto j        = i + ofst;
    const bool valid_i  = STATIC_RESULT_POSITION ? (j < num_k) : (j < max_result_position);
    const auto child_id = valid_i ? result_child_indices_ptr[j] : invalid_index;

    // CRITICAL: ALL threads in the team must participate in compute_distance and team_sum
    // Otherwise warp shuffles will hang. Each thread calls the unified extern function to get
    // its per-thread distance, then team_sum reduces across all threads in the team.
    DistanceT per_thread_dist = 0;
    if (child_id != invalid_index) {
      // Use unified compute_distance function (links standard or VPQ fragment at runtime)
      per_thread_dist = compute_distance<TeamSize,
                                         DatasetBlockDim,
                                         PQ_BITS,
                                         PQ_LEN,
                                         CodebookT,
                                         DataT,
                                         IndexT,
                                         DistanceT,
                                         QueryT>(args, child_id);
    } else {
      // Invalid child_id: lead lane gets upper_bound, others get 0
      per_thread_dist = lead_lane ? raft::upper_bound<DistanceT>() : 0;
    }

    // Now ALL threads in the team participate in team_sum
    DistanceT child_dist = device::team_sum(per_thread_dist, team_size_bits);
    __syncwarp();

    // Store the distance
    if (valid_i && lead_lane) { result_child_distances_ptr[j] = child_dist; }
  }
}

}  // namespace device
}  // namespace cuvs::neighbors::cagra::detail
