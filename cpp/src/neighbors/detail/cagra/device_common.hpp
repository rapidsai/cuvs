/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

template <uint32_t Dim = 1024, uint32_t Stride = 128, typename T>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto swizzling(T x) -> T
{
  // Address swizzling reduces bank conflicts in shared memory, but increases
  // the amount of operation instead.
  // return x;
  if constexpr (Stride <= 32) {
    return x;
  } else if constexpr (Dim <= 1024) {
    return x ^ (x >> 5);
  } else {
    return x ^ ((x >> 5) & 0x1f);
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
RAFT_DEVICE_INLINE_FUNCTION auto team_sum(T x, uint32_t team_size_bitshift) -> T
{
  switch (team_size_bitshift) {
    case 5: x += raft::shfl_xor(x, 16);
    case 4: x += raft::shfl_xor(x, 8);
    case 3: x += raft::shfl_xor(x, 4);
    case 2: x += raft::shfl_xor(x, 2);
    case 1: x += raft::shfl_xor(x, 1);
    default: return x;
  }
}

template <typename IndexT,
          typename DistanceT,
          typename DATASET_DESCRIPTOR_T>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_nodes(
  IndexT* __restrict__ result_indices_ptr,       // [num_pickup]
  DistanceT* __restrict__ result_distances_ptr,  // [num_pickup]
  const DATASET_DESCRIPTOR_T& dataset_desc,
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
  const auto team_size_bits = dataset_desc.team_size_bitshift_from_smem();
  const auto max_i = raft::round_up_safe<uint32_t>(num_pickup, warp_size >> team_size_bits);
  const auto compute_distance = dataset_desc.compute_distance_impl;

  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += (blockDim.x >> team_size_bits)) {
    const bool valid_i = (i < num_pickup);

    IndexT best_index_team_local    = raft::upper_bound<IndexT>();
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

      const auto norm2 = dataset_desc.compute_distance(seed_index, valid_i);

      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
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

template <typename IndexT,
          typename DistanceT,
          typename DATASET_DESCRIPTOR_T,
          int STATIC_RESULT_POSITION = 1>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_child_nodes(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  // [dataset_dim, dataset_size]
  const DATASET_DESCRIPTOR_T& dataset_desc,
  // [knn_k, dataset_size]
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  // hashmap
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
      if (hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child_id) == 0) {
        // Deactivate this entry as insertion into visited hash table has failed.
        child_id = invalid_index;
      } else if ((traversed_hashmap_ptr != nullptr) &&
                 hashmap::search<IndexT, 1>(
                   traversed_hashmap_ptr, traversed_hash_bitlen, child_id)) {
        // Deactivate this entry as this has been already used by others.
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

  // Compute the distance to child nodes
  const auto team_size_bits   = dataset_desc.team_size_bitshift_from_smem();
  const auto num_k            = knn_k * search_width;
  const auto max_i            = raft::round_up_safe(num_k, warp_size >> team_size_bits);
  const auto compute_distance = dataset_desc.compute_distance_impl;
  const auto args             = dataset_desc.args.load();
  const bool lead_lane        = (threadIdx.x & ((1u << team_size_bits) - 1u)) == 0;
  const uint32_t ofst         = STATIC_RESULT_POSITION ? 0 : result_position[0];
  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += blockDim.x >> team_size_bits) {
    const auto j        = i + ofst;
    const bool valid_i  = STATIC_RESULT_POSITION ? (j < num_k) : (j < max_result_position);
    const auto child_id = valid_i ? result_child_indices_ptr[j] : invalid_index;

    // We should be calling `dataset_desc.compute_distance(..)` here as follows:
    // > const auto child_dist = dataset_desc.compute_distance(child_id, child_id != invalid_index);
    // Instead, we manually inline this function for performance reasons.
    // This allows us to move the fetching of the arguments from shared memory out of the loop.
    const DistanceT child_dist = device::team_sum(
      (child_id != invalid_index) ? compute_distance(args, child_id)
                                  : (lead_lane ? raft::upper_bound<DistanceT>() : 0),
      team_size_bits);
    __syncwarp();

    // Store the distance
    if (valid_i && lead_lane) { result_child_distances_ptr[j] = child_dist; }
  }
}

RAFT_DEVICE_INLINE_FUNCTION void lds(float& x, uint32_t addr)
{
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half& x, uint32_t addr)
{
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(reinterpret_cast<uint16_t&>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half2& x, uint32_t addr)
{
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(reinterpret_cast<uint32_t&>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[1], uint32_t addr)
{
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*reinterpret_cast<uint16_t*>(x)) : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[2], uint32_t addr)
{
  asm volatile("ld.shared.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "r"(addr));
}
RAFT_DEVICE_INLINE_FUNCTION void lds(half (&x)[4], uint32_t addr)
{
  asm volatile("ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint8_t& x, uint32_t addr)
{
  uint32_t res;
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(res) : "r"(addr));
  x = static_cast<uint32_t>(res);
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint32_t& x, uint32_t addr)
{
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint32_t& x, const uint32_t* addr)
{
  lds(x, uint32_t(__cvta_generic_to_shared(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint4& x, uint32_t addr)
{
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(uint4& x, const uint4* addr)
{
  lds(x, uint32_t(__cvta_generic_to_shared(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void lds(int8_t& x, uint32_t addr)
{
  int32_t res;
  asm volatile("ld.shared.s8 {%0}, [%1];" : "=r"(res) : "r"(addr));
  x = static_cast<int8_t>(res);
}

RAFT_DEVICE_INLINE_FUNCTION void lds(int4& x, uint32_t addr)
{
  asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "r"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void sts(uint32_t addr, const half2& x)
{
  asm volatile("st.shared.v2.u16 [%0], {%1, %2};"
               :
               : "r"(addr),
                 "h"(reinterpret_cast<const uint16_t&>(x.x)),
                 "h"(reinterpret_cast<const uint16_t&>(x.y)));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_cg(uint4& x, const uint4* addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(uint4& x, const uint4* addr)
{
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(uint32_t& x, const uint32_t* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_cg(uint32_t& x, const uint32_t* addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half& x, const half* addr)
{
  asm volatile("ld.global.ca.u16 {%0}, [%1];"
               : "=h"(reinterpret_cast<uint16_t&>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[1], const half* addr)
{
  asm volatile("ld.global.ca.u16 {%0}, [%1];"
               : "=h"(*reinterpret_cast<uint16_t*>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[2], const half* addr)
{
  asm volatile("ld.global.ca.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half (&x)[4], const half* addr)
{
  asm volatile("ld.global.ca.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}

RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2& x, const half* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(reinterpret_cast<uint32_t&>(x))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2 (&x)[1], const half* addr)
{
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(*reinterpret_cast<uint32_t*>(x))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}
RAFT_DEVICE_INLINE_FUNCTION void ldg_ca(half2 (&x)[2], const half* addr)
{
  asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];"
               : "=r"(*reinterpret_cast<uint32_t*>(x)), "=r"(*reinterpret_cast<uint32_t*>(x + 1))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
}

}  // namespace device
}  // namespace cuvs::neighbors::cagra::detail
