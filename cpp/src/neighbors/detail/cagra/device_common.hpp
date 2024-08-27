/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "hashmap.hpp"
#include "utils.hpp"

#include <cuvs/distance/distance.hpp>

// TODO: This shouldn't be invoking anything in detail APIs outside of cuvs/neighbors
#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/warp_primitives.cuh>

#include <cuda_fp16.h>

#include <cfloat>
#include <cstdint>

namespace cuvs::neighbors::cagra::detail {
namespace device {

// warpSize for compile time calculation
constexpr unsigned warp_size = 32;

// using LOAD_256BIT_T = ulonglong4;
using LOAD_128BIT_T = uint4;
using LOAD_64BIT_T  = uint64_t;

template <class LOAD_T, class DATA_T>
RAFT_DEVICE_INLINE_FUNCTION constexpr unsigned get_vlen()
{
  return utils::size_of<LOAD_T>() / utils::size_of<DATA_T>();
}

/** Xorshift rondem number generator.
 *
 * See https://en.wikipedia.org/wiki/Xorshift#xorshift for reference.
 */
_RAFT_HOST_DEVICE inline uint64_t xorshift64(uint64_t u)
{
  u ^= u >> 12;
  u ^= u << 25;
  u ^= u >> 27;
  return u * 0x2545F4914F6CDD1DULL;
}

template <class T, unsigned X_MAX = 1024>
RAFT_DEVICE_INLINE_FUNCTION constexpr T swizzling(T x)
{
  // Address swizzling reduces bank conflicts in shared memory, but increases
  // the amount of operation instead.
  // return x;
  if constexpr (X_MAX <= 1024) {
    return (x) ^ ((x) >> 5);
  } else {
    return (x) ^ (((x) >> 5) & 0x1f);
  }
}

template <typename IndexT,
          typename DistanceT,
          typename DATASET_DESCRIPTOR_T>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_nodes(
  IndexT* __restrict__ result_indices_ptr,       // [num_pickup]
  DistanceT* __restrict__ result_distances_ptr,  // [num_pickup]
  const DATASET_DESCRIPTOR_T& __restrict__ dataset_desc,
  const size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* __restrict__ seed_ptr,  // [num_seeds]
  const uint32_t num_seeds,
  IndexT* __restrict__ visited_hash_ptr,
  const uint32_t hash_bitlen,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1)
{
  const auto team_size = dataset_desc.team_size;
  uint32_t max_i       = num_pickup;
  if (max_i % (warp_size / team_size)) {
    max_i += (warp_size / team_size) - (max_i % (warp_size / team_size));
  }

  for (uint32_t i = threadIdx.x / team_size; i < max_i; i += blockDim.x / team_size) {
    const bool valid_i = (i < num_pickup);

    IndexT best_index_team_local;
    DistanceT best_norm2_team_local = raft::upper_bound<DistanceT>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      // Select a node randomly and compute the distance to it
      IndexT seed_index;
      if (valid_i) {
        // uint32_t gid = i + (num_pickup * (j + (num_distilation * block_id)));
        uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % dataset_desc.size;
        }
      }

      auto norm2 = dataset_desc.compute_distance(seed_index, valid_i);

      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x % team_size;
    if (valid_i && lane_id == 0) {
      if (hashmap::insert(visited_hash_ptr, hash_bitlen, best_index_team_local)) {
        result_distances_ptr[i] = best_norm2_team_local;
        result_indices_ptr[i]   = best_index_team_local;
      } else {
        result_distances_ptr[i] = raft::upper_bound<DistanceT>();
        result_indices_ptr[i]   = raft::upper_bound<IndexT>();
      }
    }
  }
}

template <typename IndexT, typename DistanceT, typename DATASET_DESCRIPTOR_T>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_child_nodes(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  // [dataset_dim, dataset_size]
  const DATASET_DESCRIPTOR_T& __restrict__ dataset_desc,
  // [knn_k, dataset_size]
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  // hashmap
  IndexT* __restrict__ visited_hashmap_ptr,
  const uint32_t hash_bitlen,
  const IndexT* __restrict__ parent_indices,
  const IndexT* __restrict__ internal_topk_list,
  const uint32_t search_width)
{
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  constexpr IndexT invalid_index    = raft::upper_bound<IndexT>();

  // Read child indices of parents from knn graph and check if the distance
  // computaiton is necessary.
  for (uint32_t i = threadIdx.x; i < knn_k * search_width; i += blockDim.x) {
    const IndexT smem_parent_id = parent_indices[i / knn_k];
    IndexT child_id             = invalid_index;
    if (smem_parent_id != invalid_index) {
      const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
      child_id             = knn_graph[(i % knn_k) + (static_cast<int64_t>(knn_k) * parent_id)];
    }
    if (child_id != invalid_index) {
      if (hashmap::insert(visited_hashmap_ptr, hash_bitlen, child_id) == 0) {
        child_id = invalid_index;
      }
    }
    result_child_indices_ptr[i] = child_id;
  }
  __syncthreads();

  // Compute the distance to child nodes
  uint32_t max_i       = knn_k * search_width;
  const auto team_size = dataset_desc.team_size;
  if (max_i % (warp_size / team_size)) {
    max_i += (warp_size / team_size) - (max_i % (warp_size / team_size));
  }
  for (uint32_t tid = threadIdx.x; tid < max_i * team_size; tid += blockDim.x) {
    const auto i       = tid / team_size;
    const bool valid_i = (i < (knn_k * search_width));
    IndexT child_id    = invalid_index;
    if (valid_i) { child_id = result_child_indices_ptr[i]; }

    auto norm2 = dataset_desc.compute_distance(child_id, child_id != invalid_index);

    // Store the distance
    const unsigned lane_id = threadIdx.x % team_size;
    if (valid_i && lane_id == 0) {
      if (child_id != invalid_index) {
        result_child_distances_ptr[i] = norm2;
      } else {
        result_child_distances_ptr[i] = raft::upper_bound<DistanceT>();
      }
    }
  }
}

template <uint32_t TeamSize, typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x) -> T
{
#pragma unroll
  for (uint32_t stride = TeamSize >> 1; stride > 0; stride >>= 1) {
    x += raft::shfl_xor(x, stride, TeamSize);
  }
  return x;
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x, uint32_t team_size) -> T
{
  switch (team_size) {
    case 32: x += raft::shfl_xor(x, 16);
    case 16: x += raft::shfl_xor(x, 8);
    case 8: x += raft::shfl_xor(x, 4);
    case 4: x += raft::shfl_xor(x, 2);
    case 2: x += raft::shfl_xor(x, 1);
    default: return x;
  }
}

}  // namespace device
}  // namespace cuvs::neighbors::cagra::detail
