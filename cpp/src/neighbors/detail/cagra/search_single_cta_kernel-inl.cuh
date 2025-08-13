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

#include "search_single_cta_kernel.cuh"

#include "bitonic.hpp"
#include "compute_distance-ext.cuh"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_by_radix.cuh"
#include "topk_for_cagra/topk.h"  // TODO replace with raft topk
#include "utils.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/neighbors/common.hpp>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
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
#include <stdint.h>
#include <thread>
#include <vector>

namespace cuvs::neighbors::cagra::detail {
namespace single_cta_search {

// #define _CLK_BREAKDOWN

template <unsigned TOPK_BY_BITONIC_SORT, class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void pickup_next_parents(std::uint32_t* const terminate_flag,
                                                     INDEX_T* const next_parent_indices,
                                                     INDEX_T* const internal_topk_indices,
                                                     const std::size_t internal_topk_size,
                                                     const std::uint32_t search_width)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  // if (threadIdx.x >= 32) return;

  for (std::uint32_t i = threadIdx.x; i < search_width; i += 32) {
    next_parent_indices[i] = utils::get_max_value<INDEX_T>();
  }
  std::uint32_t itopk_max = internal_topk_size;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }
  std::uint32_t num_new_parents = 0;
  for (std::uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    std::uint32_t jj = j;
    if (TOPK_BY_BITONIC_SORT) { jj = device::swizzling(j); }
    INDEX_T index;
    int new_parent = 0;
    if (j < internal_topk_size) {
      index = internal_topk_indices[jj];
      if ((index & index_msb_1_mask) == 0) {  // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
      if (i < search_width) {
        next_parent_indices[i] = jj;
        // set most significant bit as used node
        internal_topk_indices[jj] |= index_msb_1_mask;
      }
    }
    num_new_parents += __popc(ballot_mask);
    if (num_new_parents >= search_width) { break; }
  }
  if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}

template <unsigned MAX_CANDIDATES, class IdxT = void>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_full(
  float* candidate_distances,  // [num_candidates]
  IdxT* candidate_indices,     // [num_candidates]
  const std::uint32_t num_candidates,
  const std::uint32_t num_itopk,
  unsigned MULTI_WARPS = 0)
{
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;
  if (MULTI_WARPS == 0) {
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_CANDIDATES + 31) / 32;
    float key[N];
    IdxT val[N];
    /* Candidates -> Reg */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = lane_id + (32 * i);
      if (j < num_candidates) {
        key[i] = candidate_distances[j];
        val[i] = candidate_indices[j];
      } else {
        key[i] = utils::get_max_value<float>();
        val[i] = utils::get_max_value<IdxT>();
      }
    }
    /* Sort */
    bitonic::warp_sort<float, IdxT, N>(key, val);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_candidates && j < num_itopk) {
        candidate_distances[device::swizzling(j)] = key[i];
        candidate_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    // Use two warps (64 threads)
    constexpr unsigned max_candidates_per_warp = (MAX_CANDIDATES + 1) / 2;
    constexpr unsigned N                       = (max_candidates_per_warp + 31) / 32;
    float key[N];
    IdxT val[N];
    if (warp_id < 2) {
      /* Candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = lane_id + (32 * i);
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates) {
          key[i] = candidate_distances[j];
          val[i] = candidate_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Sort */
      bitonic::warp_sort<float, IdxT, N>(key, val);
      /* Reg -> Temp_candidates */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && jl < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    __syncthreads();

    unsigned num_warps_used = (num_itopk + max_candidates_per_warp - 1) / max_candidates_per_warp;
    if (warp_id < num_warps_used) {
      /* Temp_candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned kl = max_candidates_per_warp - 1 - jl;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        unsigned k  = MAX_CANDIDATES - 1 - j;
        if (j >= num_candidates || k >= num_candidates || kl >= num_itopk) continue;
        float temp_key = candidate_distances[device::swizzling(k)];
        if (key[i] == temp_key) continue;
        if ((warp_id == 0) == (key[i] > temp_key)) {
          key[i] = temp_key;
          val[i] = candidate_indices[device::swizzling(k)];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
    if (warp_id < num_warps_used) {
      /* Merge */
      bitonic::warp_merge<float, IdxT, N>(key, val, 32);
      /* Reg -> Temp_itopk */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && j < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
  }
}

template <unsigned MAX_ITOPK, class IdxT = void>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_merge(
  float* itopk_distances,  // [num_itopk]
  IdxT* itopk_indices,     // [num_itopk]
  const std::uint32_t num_itopk,
  float* candidate_distances,  // [num_candidates]
  IdxT* candidate_indices,     // [num_candidates]
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first,
  unsigned MULTI_WARPS = 0)
{
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;
  if (MULTI_WARPS == 0) {
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_ITOPK + 31) / 32;
    float key[N];
    IdxT val[N];
    if (first) {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = lane_id + (32 * i);
        if (j < num_itopk) {
          key[i] = itopk_distances[j];
          val[i] = itopk_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Warp Sort */
      bitonic::warp_sort<float, IdxT, N>(key, val);
    } else {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = (N * lane_id) + i;
        if (j < num_itopk) {
          key[i] = itopk_distances[device::swizzling(j)];
          val[i] = itopk_indices[device::swizzling(j)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
    }
    /* Merge candidates */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;  // [0:MAX_ITOPK-1]
      unsigned k = MAX_ITOPK - 1 - j;
      if (k >= num_itopk || k >= num_candidates) continue;
      float candidate_key = candidate_distances[device::swizzling(k)];
      if (key[i] > candidate_key) {
        key[i] = candidate_key;
        val[i] = candidate_indices[device::swizzling(k)];
      }
    }
    /* Warp Merge */
    bitonic::warp_merge<float, IdxT, N>(key, val, 32);
    /* Store new itopk results */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_itopk) {
        itopk_distances[device::swizzling(j)] = key[i];
        itopk_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    // Use two warps (64 threads) or more
    constexpr unsigned max_itopk_per_warp = (MAX_ITOPK + 1) / 2;
    constexpr unsigned N                  = (max_itopk_per_warp + 31) / 32;
    float key[N];
    IdxT val[N];
    if (first) {
      /* Load itop results (not sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = lane_id + (32 * i) + (max_itopk_per_warp * warp_id);
          if (j < num_itopk) {
            key[i] = itopk_distances[j];
            val[i] = itopk_indices[j];
          } else {
            key[i] = utils::get_max_value<float>();
            val[i] = utils::get_max_value<IdxT>();
          }
        }
        /* Warp Sort */
        bitonic::warp_sort<float, IdxT, N>(key, val);
        /* Store intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
      __syncthreads();
      if (warp_id < 2) {
        /* Load intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          unsigned k = MAX_ITOPK - 1 - j;
          if (k >= num_itopk) continue;
          float temp_key = itopk_distances[device::swizzling(k)];
          if (key[i] == temp_key) continue;
          if ((warp_id == 0) == (key[i] > temp_key)) {
            key[i] = temp_key;
            val[i] = itopk_indices[device::swizzling(k)];
          }
        }
        /* Warp Merge */
        bitonic::warp_merge<float, IdxT, N>(key, val, 32);
      }
      __syncthreads();
      /* Store itopk results (sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    const uint32_t num_itopk_div2 = num_itopk / 2;
    if (threadIdx.x < 3) {
      // work_buf is used to obtain turning points in 1st and 2nd half of itopk afer merge.
      work_buf[threadIdx.x] = num_itopk_div2;
    }
    __syncthreads();

    // Merge candidates (using whole threads)
    for (unsigned k = threadIdx.x; k < min(num_candidates, num_itopk); k += blockDim.x) {
      const unsigned j          = num_itopk - 1 - k;
      const float itopk_key     = itopk_distances[device::swizzling(j)];
      const float candidate_key = candidate_distances[device::swizzling(k)];
      if (itopk_key > candidate_key) {
        itopk_distances[device::swizzling(j)] = candidate_key;
        itopk_indices[device::swizzling(j)]   = candidate_indices[device::swizzling(k)];
        if (j < num_itopk_div2) {
          atomicMin(work_buf + 2, j);
        } else {
          atomicMin(work_buf + 1, j - num_itopk_div2);
        }
      }
    }
    __syncthreads();

    // Merge 1st and 2nd half of itopk (using whole threads)
    for (unsigned j = threadIdx.x; j < num_itopk_div2; j += blockDim.x) {
      const unsigned k = j + num_itopk_div2;
      float key_0      = itopk_distances[device::swizzling(j)];
      float key_1      = itopk_distances[device::swizzling(k)];
      if (key_0 > key_1) {
        itopk_distances[device::swizzling(j)] = key_1;
        itopk_distances[device::swizzling(k)] = key_0;
        IdxT val_0                            = itopk_indices[device::swizzling(j)];
        IdxT val_1                            = itopk_indices[device::swizzling(k)];
        itopk_indices[device::swizzling(j)]   = val_1;
        itopk_indices[device::swizzling(k)]   = val_0;
        atomicMin(work_buf + 0, j);
      }
    }
    if (threadIdx.x == blockDim.x - 1) {
      if (work_buf[2] < num_itopk_div2) { work_buf[1] = work_buf[2]; }
    }
    __syncthreads();
    // if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    //     RAFT_LOG_DEBUG( "work_buf: %u, %u, %u\n", work_buf[0], work_buf[1], work_buf[2] );
    // }

    // Warp-0 merges 1st half of itopk, warp-1 does 2nd half.
    if (warp_id < 2) {
      // Load intermedidate itopk results
      const uint32_t turning_point = work_buf[warp_id];  // turning_point <= num_itopk_div2
      for (unsigned i = 0; i < N; i++) {
        unsigned k = num_itopk;
        unsigned j = (N * lane_id) + i;
        if (j < turning_point) {
          k = j + (num_itopk_div2 * warp_id);
        } else if (j >= (MAX_ITOPK / 2 - num_itopk_div2)) {
          j -= (MAX_ITOPK / 2 - num_itopk_div2);
          if ((turning_point <= j) && (j < num_itopk_div2)) { k = j + (num_itopk_div2 * warp_id); }
        }
        if (k < num_itopk) {
          key[i] = itopk_distances[device::swizzling(k)];
          val[i] = itopk_indices[device::swizzling(k)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<IdxT>();
        }
      }
      /* Warp Merge */
      bitonic::warp_merge<float, IdxT, N>(key, val, 32);
      /* Store new itopk results */
      for (unsigned i = 0; i < N; i++) {
        const unsigned j = (N * lane_id) + i;
        if (j < num_itopk_div2) {
          unsigned k                            = j + (num_itopk_div2 * warp_id);
          itopk_distances[device::swizzling(k)] = key[i];
          itopk_indices[device::swizzling(k)]   = val[i];
        }
      }
    }
  }
}

template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          class IdxT>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort_and_merge(
  float* itopk_distances,  // [num_itopk]
  IdxT* itopk_indices,     // [num_itopk]
  const std::uint32_t num_itopk,
  float* candidate_distances,  // [num_candidates]
  IdxT* candidate_indices,     // [num_candidates]
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first,
  const unsigned MULTI_WARPS_1,
  const unsigned MULTI_WARPS_2)
{
  // The results in candidate_distances/indices are sorted by bitonic sort.
  topk_by_bitonic_sort_and_full<MAX_CANDIDATES, IdxT>(
    candidate_distances, candidate_indices, num_candidates, num_itopk, MULTI_WARPS_1);

  // The results sorted above are merged with the internal intermediate top-k
  // results so far using bitonic merge.
  topk_by_bitonic_sort_and_merge<MAX_ITOPK, IdxT>(itopk_distances,
                                                  itopk_indices,
                                                  num_itopk,
                                                  candidate_distances,
                                                  candidate_indices,
                                                  num_candidates,
                                                  work_buf,
                                                  first,
                                                  MULTI_WARPS_2);
}

// This function move the invalid index element to the end of the itopk list.
// Require : array_length % 32 == 0 && The invalid entry is only one.
template <class IdxT>
RAFT_DEVICE_INLINE_FUNCTION void move_invalid_to_end_of_list(IdxT* const index_array,
                                                             float* const distance_array,
                                                             const std::uint32_t array_length)
{
  constexpr std::uint32_t warp_size     = 32;
  constexpr std::uint32_t invalid_index = utils::get_max_value<IdxT>();
  const std::uint32_t lane_id           = threadIdx.x % warp_size;

  if (threadIdx.x >= warp_size) { return; }

  bool found_invalid = false;
  if (array_length % warp_size == 0) {
    for (std::uint32_t i = lane_id; i < array_length; i += warp_size) {
      const auto index    = index_array[i];
      const auto distance = distance_array[i];

      if (found_invalid) {
        index_array[i - 1]    = index;
        distance_array[i - 1] = distance;
      } else {
        // Check if the index is invalid
        const auto I_found_invalid = (index == invalid_index);
        const auto who_has_invalid = raft::ballot(I_found_invalid);
        // if a value that is loaded by a smaller lane id thread, shift the array
        if (who_has_invalid << (warp_size - lane_id)) {
          index_array[i - 1]    = index;
          distance_array[i - 1] = distance;
        }

        found_invalid = who_has_invalid;
      }
    }
  }
  if (lane_id == 0) {
    index_array[array_length - 1]    = invalid_index;
    distance_array[array_length - 1] = utils::get_max_value<float>();
  }
}

template <class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void hashmap_restore(INDEX_T* const hashmap_ptr,
                                                 const size_t hashmap_bitlen,
                                                 const INDEX_T* itopk_indices,
                                                 const uint32_t itopk_size,
                                                 const uint32_t first_tid = 0)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  if (threadIdx.x < first_tid) return;
  for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
    auto key = itopk_indices[i] & ~index_msb_1_mask;  // clear most significant bit
    hashmap::insert(hashmap_ptr, hashmap_bitlen, key);
  }
}

/**
 * @brief Search operation for a single query using a single thread block.
 * *
 * @tparam MAX_ITOPK Maximum for the internal_topk argument.
 * @tparam MAX_CANDIDATES
 * @tparam TOPK_BY_BITONIC_SORT
 * @tparam DATASET_DESCRIPTOR_T
 * @tparam SAMPLE_FILTER_T
 *
 * @param result_indices_ptr
 *    Tagged pointer to the result neighbors [num_queries, top_k]; the tag is the two lower bits to
 *    identify the index element type (see the code below).
 * @param result_distances_ptr Pointer to the result distances buffer [num_queries, top_k].
 * @param top_k Number of top-k results to retrieve.
 * @param dataset_desc Pointer to the dataset descriptor.
 * @param queries_ptr Pointer to the queries [num_queries, dataset_dim].
 * @param knn_graph Pointer to the k-nearest neighbors graph [dataset_size, graph_degree].
 * @param graph_degree Degree of the graph.
 * @param num_distilation Number of distillation steps.
 * @param rand_xor_mask Random XOR mask for randomization.
 * @param seed_ptr Pointer to the seed indices [num_queries, num_seeds].
 * @param num_seeds Number of seeds.
 * @param visited_hashmap_ptr
 *    Pointer to the hashmap of visited nodes [num_queries, 1 << hash_bitlen].
 * @param internal_topk Internal top-k size.
 * @param search_width Width of the search.
 * @param min_iteration Minimum number of iterations.
 * @param max_iteration Maximum number of iterations.
 * @param num_executed_iterations Pointer to the number of executed iterations [num_queries].
 * @param hash_bitlen Bit length of the hash.
 * @param small_hash_bitlen Bit length of the small hash.
 * @param small_hash_reset_interval Interval for resetting the small hash.
 * @param query_id sequential id of the query in the batch
 */
template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
__device__ void search_core(
  uintptr_t result_indices_ptr,                                           // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id,
  SAMPLE_FILTER_T sample_filter)
{
  using LOAD_T = device::LOAD_128BIT_T;

  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

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
  // +----------------------+------------------------------+---------+
  // | internal_top_k       | neighbors of internal_top_k  | padding |
  // | <internal_topk_size> | <search_width * graph_degree> | upto 32 |
  // +----------------------+------------------------------+---------+
  // |<---             result_buffer_size              --->|
  const auto result_buffer_size    = internal_topk + (search_width * graph_degree);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  const auto small_hash_size       = hashmap::get_size(small_hash_bitlen);

  // Set smem working buffer for the distance calculation
  dataset_desc = dataset_desc->setup_workspace(smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + dataset_desc->smem_ws_size_in_bytes());
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ visited_hash_buffer =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_list_buffer =
    reinterpret_cast<INDEX_T*>(visited_hash_buffer + small_hash_size);
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
  INDEX_T* local_visited_hashmap_ptr;
  if (small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * blockIdx.y);
  }
  hashmap::init(local_visited_hashmap_ptr, hash_bitlen, 0);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes
  _CLK_START();
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  device::compute_distance_to_random_nodes(result_indices_buffer,
                                           result_distances_buffer,
                                           *dataset_desc,
                                           result_buffer_size,
                                           num_distilation,
                                           rand_xor_mask,
                                           local_seed_ptr,
                                           num_seeds,
                                           local_visited_hashmap_ptr,
                                           hash_bitlen,
                                           (INDEX_T*)nullptr,
                                           0);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  std::uint32_t iter = 0;
  while (1) {
    // sort
    if constexpr (TOPK_BY_BITONIC_SORT) {
      // [Notice]
      // It is good to use multiple warps in topk_by_bitonic_sort_and_merge() when
      // batch size is small (short-latency), but it might not be always good
      // when batch size is large (high-throughput).
      // topk_by_bitonic_sort_and_merge() consists of two operations:
      // if MAX_CANDIDATES is greater than 128, the first operation uses two warps;
      // if MAX_ITOPK is greater than 256, the second operation used two warps.
      const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;

      // reset small-hash table.
      if ((iter + 1) % small_hash_reset_interval == 0) {
        // Depending on the block size and the number of warps used in
        // topk_by_bitonic_sort_and_merge(), determine which warps are used to reset
        // the small hash and whether they are performed in overlap with
        // topk_by_bitonic_sort_and_merge().
        _CLK_START();
        unsigned hash_start_tid;
        if (blockDim.x == 32) {
          hash_start_tid = 0;
        } else if (blockDim.x == 64) {
          if (multi_warps_1 || multi_warps_2) {
            hash_start_tid = 0;
          } else {
            hash_start_tid = 32;
          }
        } else {
          if (multi_warps_1 || multi_warps_2) {
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
      if (!(std::is_same<SAMPLE_FILTER_T, cuvs::neighbors::filtering::none_sample_filter>::value ||
            *filter_flag == 0)) {
        // Move the filtered out index to the end of the itopk list
        for (unsigned i = 0; i < search_width; i++) {
          move_invalid_to_end_of_list(
            result_indices_buffer, result_distances_buffer, internal_topk);
        }

        if (threadIdx.x == 0) { *terminate_flag = 0; }
      }
      topk_by_bitonic_sort_and_merge<MAX_ITOPK, MAX_CANDIDATES>(
        result_distances_buffer,
        result_indices_buffer,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        search_width * graph_degree,
        topk_ws,
        (iter == 0),
        multi_warps_1,
        multi_warps_2);
      __syncthreads();
      _CLK_REC(clk_topk);
    } else {
      _CLK_START();
      // topk with radix block sort
      topk_by_radix_sort<MAX_ITOPK, INDEX_T>{}(
        internal_topk,
        gridDim.x,
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
      pickup_next_parents<TOPK_BY_BITONIC_SORT, INDEX_T>(
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

    // compute the norms between child nodes and query node
    _CLK_START();
    device::compute_distance_to_child_nodes(result_indices_buffer + internal_topk,
                                            result_distances_buffer + internal_topk,
                                            *dataset_desc,
                                            knn_graph,
                                            graph_degree,
                                            local_visited_hashmap_ptr,
                                            hash_bitlen,
                                            (INDEX_T*)nullptr,
                                            0,
                                            parent_list_buffer,
                                            result_indices_buffer,
                                            search_width);
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                cuvs::neighbors::filtering::none_sample_filter>::value) {
      if (threadIdx.x == 0) { *filter_flag = 0; }
      __syncthreads();

      constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
      const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

      for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
        if (parent_list_buffer[p] != invalid_index) {
          const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter(query_id, parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
            *filter_flag                                   = 1;
          }
        }
      }
      __syncthreads();
    }

    iter++;
  }

  // Post process for filtering
  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              cuvs::neighbors::filtering::none_sample_filter>::value) {
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
    const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

    for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree;
         i += blockDim.x) {
      const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
      if (node_id != (invalid_index & ~index_msb_1_mask) && !sample_filter(query_id, node_id)) {
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
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
        // Calculate the new buffer index
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

        // Calculate the largest valid position within a warp and bcast it for the next iteration
        num_found_valid += new_position;
        for (std::uint32_t offset = (warp_size >> 1); offset > 0; offset >>= 1) {
          const auto v = raft::shfl_xor(num_found_valid, offset);
          if ((threadIdx.x & offset) == 0) { num_found_valid = v; }
        }

        // If the enough number of items are found, do early termination
        if (num_found_valid >= top_k) { break; }
      }

      if (num_found_valid < top_k) {
        // Fill the remaining buffer with invalid values so that `topk_by_bitonic_sort_and_merge` is
        // usable in the next step
        for (std::uint32_t i = num_found_valid + threadIdx.x; i < internal_topk; i += warp_size) {
          result_indices_buffer[i]   = invalid_index;
          result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
        }
      }
    }

    // If the sufficient number of valid indexes are not in the internal topk, pick up from the
    // candidate list.
    if (top_k > internal_topk || result_indices_buffer[top_k - 1] == invalid_index) {
      __syncthreads();
      const unsigned multi_warps_1 = ((blockDim.x >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      const unsigned multi_warps_2 = ((blockDim.x >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;
      topk_by_bitonic_sort_and_merge<MAX_ITOPK, MAX_CANDIDATES>(
        result_distances_buffer,
        result_indices_buffer,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        search_width * graph_degree,
        topk_ws,
        (iter == 0),
        multi_warps_1,
        multi_warps_2);
    }
    __syncthreads();
  }

  // NB: The indices pointer is tagged with its element size.
  //     Here we select the correct conversion operator at runtime.
  //     This allows us to avoid multiplying kernel instantiations
  //     and any costs for extra registers in the kernel signature.
  const uint32_t index_element_tag = result_indices_ptr & 0x3;
  result_indices_ptr ^= index_element_tag;
  auto write_indices =
    index_element_tag == 3
      ? [](uintptr_t ptr,
           uint32_t i,
           INDEX_T x) { reinterpret_cast<uint64_t*>(ptr)[i] = static_cast<uint64_t>(x); }
    : index_element_tag == 2
      ? [](uintptr_t ptr,
           uint32_t i,
           INDEX_T x) { reinterpret_cast<uint32_t*>(ptr)[i] = static_cast<uint32_t>(x); }
    : index_element_tag == 1
      ? [](uintptr_t ptr,
           uint32_t i,
           INDEX_T x) { reinterpret_cast<uint16_t*>(ptr)[i] = static_cast<uint16_t>(x); }
      : [](uintptr_t ptr, uint32_t i, INDEX_T x) {
          reinterpret_cast<uint8_t*>(ptr)[i] = static_cast<uint8_t>(x);
        };
  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }
    constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

    write_indices(result_indices_ptr,
                  j,
                  result_indices_buffer[ii] & ~index_msb_1_mask);  // clear most significant bit
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

template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
RAFT_KERNEL __launch_bounds__(1024, 1) search_kernel(
  uintptr_t result_indices_ptr,                                           // [num_queries, top_k]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
  const std::uint32_t top_k,
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  SAMPLE_FILTER_T sample_filter)
{
  const auto query_id = blockIdx.y;
  search_core<MAX_ITOPK,
              MAX_CANDIDATES,
              TOPK_BY_BITONIC_SORT,
              DATASET_DESCRIPTOR_T,
              SAMPLE_FILTER_T>(result_indices_ptr,
                               result_distances_ptr,
                               top_k,
                               dataset_desc,
                               queries_ptr,
                               knn_graph,
                               graph_degree,
                               num_distilation,
                               rand_xor_mask,
                               seed_ptr,
                               num_seeds,
                               visited_hashmap_ptr,
                               internal_topk,
                               search_width,
                               min_iteration,
                               max_iteration,
                               num_executed_iterations,
                               hash_bitlen,
                               small_hash_bitlen,
                               small_hash_reset_interval,
                               query_id,
                               sample_filter);
}

// To make sure we avoid false sharing on both CPU and GPU, we enforce cache line size to the
// maximum of the two.
// This makes sync atomic significantly faster.
constexpr size_t kCacheLineBytes = 64;

constexpr uint32_t kMaxJobsNum              = 8192;
constexpr uint32_t kMaxWorkersNum           = 4096;
constexpr uint32_t kMaxWorkersPerThread     = 256;
constexpr uint32_t kSoftMaxWorkersPerThread = 16;

template <typename DATASET_DESCRIPTOR_T>
struct alignas(kCacheLineBytes) job_desc_t {
  using index_type    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using distance_type = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  using data_type     = typename DATASET_DESCRIPTOR_T::DATA_T;
  // The algorithm input parameters
  struct value_t {
    uintptr_t result_indices_ptr;         // [num_queries, top_k]
    distance_type* result_distances_ptr;  // [num_queries, top_k]
    const data_type* queries_ptr;         // [num_queries, dataset_dim]
    uint32_t top_k;
    uint32_t n_queries;
  };
  using blob_elem_type = uint4;
  constexpr static inline size_t kBlobSize =
    raft::div_rounding_up_safe(sizeof(value_t), sizeof(blob_elem_type));
  // Union facilitates loading the input by a warp in a single request
  union input_t {
    blob_elem_type blob[kBlobSize];  // NOLINT
    value_t value;
  } input;
  // Last thread triggers this flag.
  cuda::atomic<bool, cuda::thread_scope_system> completion_flag;
};

struct alignas(kCacheLineBytes) worker_handle_t {
  using handle_t = uint64_t;
  struct value_t {
    uint32_t desc_id;
    uint32_t query_id;
  };
  union data_t {
    handle_t handle;
    value_t value;
  };
  cuda::atomic<data_t, cuda::thread_scope_system> data;
};
static_assert(sizeof(worker_handle_t::value_t) == sizeof(worker_handle_t::handle_t));
static_assert(
  cuda::atomic<worker_handle_t::data_t, cuda::thread_scope_system>::is_always_lock_free);

constexpr worker_handle_t::handle_t kWaitForWork = std::numeric_limits<uint64_t>::max();
constexpr worker_handle_t::handle_t kNoMoreWork  = kWaitForWork - 1;

constexpr auto is_worker_busy(worker_handle_t::handle_t h) -> bool
{
  return (h != kWaitForWork) && (h != kNoMoreWork);
}

template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
RAFT_KERNEL __launch_bounds__(1024, 1) search_kernel_p(
  const DATASET_DESCRIPTOR_T* dataset_desc,
  worker_handle_t* worker_handles,
  job_desc_t<DATASET_DESCRIPTOR_T>* job_descriptors,
  uint32_t* completion_counters,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  SAMPLE_FILTER_T sample_filter)
{
  using job_desc_type = job_desc_t<DATASET_DESCRIPTOR_T>;
  __shared__ typename job_desc_type::input_t job_descriptor;
  __shared__ worker_handle_t::data_t worker_data;

  auto& worker_handle = worker_handles[blockIdx.y].data;
  uint32_t job_ix;

  while (true) {
    // wait the writing phase
    if (threadIdx.x == 0) {
      worker_handle_t::data_t worker_data_local;
      do {
        worker_data_local = worker_handle.load(cuda::memory_order_relaxed);
      } while (worker_data_local.handle == kWaitForWork);
      if (worker_data_local.handle != kNoMoreWork) {
        worker_handle.store({kWaitForWork}, cuda::memory_order_relaxed);
      }
      job_ix = worker_data_local.value.desc_id;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      worker_data = worker_data_local;
    }
    if (threadIdx.x < raft::WarpSize) {
      // Sync one warp and copy descriptor data
      static_assert(job_desc_type::kBlobSize <= raft::WarpSize);
      job_ix = raft::shfl(job_ix, 0);
      if (threadIdx.x < job_desc_type::kBlobSize && job_ix < kMaxJobsNum) {
        job_descriptor.blob[threadIdx.x] = job_descriptors[job_ix].input.blob[threadIdx.x];
      }
    }
    __syncthreads();
    if (worker_data.handle == kNoMoreWork) { break; }

    // reading phase
    auto result_indices_ptr    = job_descriptor.value.result_indices_ptr;
    auto* result_distances_ptr = job_descriptor.value.result_distances_ptr;
    auto* queries_ptr          = job_descriptor.value.queries_ptr;
    auto top_k                 = job_descriptor.value.top_k;
    auto n_queries             = job_descriptor.value.n_queries;
    auto query_id              = worker_data.value.query_id;

    // work phase
    search_core<MAX_ITOPK,
                MAX_CANDIDATES,
                TOPK_BY_BITONIC_SORT,
                DATASET_DESCRIPTOR_T,
                SAMPLE_FILTER_T>(result_indices_ptr,
                                 result_distances_ptr,
                                 top_k,
                                 dataset_desc,
                                 queries_ptr,
                                 knn_graph,
                                 graph_degree,
                                 num_distilation,
                                 rand_xor_mask,
                                 seed_ptr,
                                 num_seeds,
                                 visited_hashmap_ptr,
                                 internal_topk,
                                 search_width,
                                 min_iteration,
                                 max_iteration,
                                 num_executed_iterations,
                                 hash_bitlen,
                                 small_hash_bitlen,
                                 small_hash_reset_interval,
                                 query_id,
                                 sample_filter);

    // make sure all writes are visible even for the host
    //     (e.g. when result buffers are in pinned memory)
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);

    // arrive to mark the end of the work phase
    __syncthreads();
    if (threadIdx.x == 0) {
      auto completed_count = atomicInc(completion_counters + job_ix, n_queries - 1) + 1;
      if (completed_count >= n_queries) {
        job_descriptors[job_ix].completion_flag.store(true, cuda::memory_order_relaxed);
      }
    }
  }
}

template <bool Persistent,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          class DATASET_DESCRIPTOR_T,
          class SAMPLE_FILTER_T>
auto dispatch_kernel = []() {
  if constexpr (Persistent) {
    return search_kernel_p<MAX_ITOPK,
                           MAX_CANDIDATES,
                           TOPK_BY_BITONIC_SORT,
                           DATASET_DESCRIPTOR_T,
                           SAMPLE_FILTER_T>;
  } else {
    return search_kernel<MAX_ITOPK,
                         MAX_CANDIDATES,
                         TOPK_BY_BITONIC_SORT,
                         DATASET_DESCRIPTOR_T,
                         SAMPLE_FILTER_T>;
  }
}();

template <bool Persistent, typename DATASET_DESCRIPTOR_T, typename SAMPLE_FILTER_T>
struct search_kernel_config {
  using kernel_t =
    decltype(dispatch_kernel<Persistent, 64, 64, 0, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>);

  template <unsigned MAX_CANDIDATES, unsigned USE_BITONIC_SORT>
  static auto choose_search_kernel(unsigned itopk_size) -> kernel_t
  {
    if (itopk_size <= 64) {
      return dispatch_kernel<Persistent,
                             64,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    } else if (itopk_size <= 128) {
      return dispatch_kernel<Persistent,
                             128,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    } else if (itopk_size <= 256) {
      return dispatch_kernel<Persistent,
                             256,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    } else if (itopk_size <= 512) {
      return dispatch_kernel<Persistent,
                             512,
                             MAX_CANDIDATES,
                             USE_BITONIC_SORT,
                             DATASET_DESCRIPTOR_T,
                             SAMPLE_FILTER_T>;
    }
    THROW("No kernel for parametels itopk_size %u, max_candidates %u", itopk_size, MAX_CANDIDATES);
  }

  static auto choose_itopk_and_mx_candidates(unsigned itopk_size,
                                             unsigned num_itopk_candidates,
                                             unsigned block_size) -> kernel_t
  {
    if (num_itopk_candidates <= 64) {
      // use bitonic sort based topk
      return choose_search_kernel<64, 1>(itopk_size);
    } else if (num_itopk_candidates <= 128) {
      return choose_search_kernel<128, 1>(itopk_size);
    } else if (num_itopk_candidates <= 256) {
      return choose_search_kernel<256, 1>(itopk_size);
    } else {
      // Radix-based topk is used
      constexpr unsigned max_candidates = 32;  // to avoid build failure
      if (itopk_size <= 256) {
        return dispatch_kernel<Persistent,
                               256,
                               max_candidates,
                               0,
                               DATASET_DESCRIPTOR_T,
                               SAMPLE_FILTER_T>;
      } else if (itopk_size <= 512) {
        return dispatch_kernel<Persistent,
                               512,
                               max_candidates,
                               0,
                               DATASET_DESCRIPTOR_T,
                               SAMPLE_FILTER_T>;
      }
    }
    THROW("No kernel for parametels itopk_size %u, num_itopk_candidates %u",
          itopk_size,
          num_itopk_candidates);
  }
};

/**
 * @brief Resource queue
 *
 * @tparam T the element type
 * @tparam Size the maximum capacity of the queue (power-of-two)
 * @tparam Empty a special element value designating an empty queue slot. NB: storing `Empty` is UB.
 *
 * A shared atomic ring buffer based queue optimized for throughput when bottlenecked on `pop`
 * operation.
 *
 * @code{.cpp}
 *   // allocate the queue
 *   resource_queue_t<int32_t, 256> resource_ids;
 *
 *   // store couple values
 *   resource_ids.push(42);
 *   resource_ids.push(7);
 *
 *   // wait to get the value from the queue
 *   auto id_x = resource_ids.pop().wait();
 *
 *   // stand in line to get the value from the queue, but don't wait
 *   auto ticket_y = resource_ids.pop();
 *   // do other stuff and check if the value is available
 *   int32_t id_y;
 *   while (!ticket_y.test(id_y)) {
 *     do_some_important_business(...);
 *     std::this_thread::sleep_for(std::chrono::microseconds(10);
 *   }
 *   // `id_y` is set by now and `ticket_y.wait()` won't block anymore
 *   assert(ticket_y.wait() == id_y);
 * @endcode
 */
template <typename T, uint32_t Size, T Empty = std::numeric_limits<T>::max()>
struct alignas(kCacheLineBytes) resource_queue_t {
  using value_type                   = T;
  static constexpr uint32_t kSize    = Size;
  static constexpr value_type kEmpty = Empty;
  static_assert(cuda::std::atomic<value_type>::is_always_lock_free,
                "The value type must be lock-free.");
  static_assert(raft::is_a_power_of_two(kSize), "The size must be a power-of-two for efficiency.");
  static constexpr uint32_t kElemsPerCacheLine =
    raft::div_rounding_up_safe<uint32_t>(kCacheLineBytes, sizeof(value_type));
  /* [Note: cache-friendly indexing]
     To avoid false sharing, the queue pushes and pops values not sequentially, but with an
     increment that is larger than the cache line size.
     Hence we introduce the `kCounterIncrement > kCacheLineBytes`.
     However, to make sure all indices are used, we choose the increment to be coprime with the
     buffer size. We also require that the buffer size is a power-of-two for two reasons:
       1) Fast modulus operation - reduces to binary `and` (with `kCounterLocMask`).
       2) Easy to ensure GCD(kCounterIncrement, kSize) == 1 by construction
          (see the definition below).
   */
  static constexpr uint32_t kCounterIncrement = raft::bound_by_power_of_two(kElemsPerCacheLine) + 1;
  static constexpr uint32_t kCounterLocMask   = kSize - 1;
  // These props hold by design, but we add them here as a documentation and a sanity check.
  static_assert(
    kCounterIncrement * sizeof(value_type) >= kCacheLineBytes,
    "The counter increment should be larger than the cache line size to avoid false sharing.");
  static_assert(
    std::gcd(kCounterIncrement, kSize) == 1,
    "The counter increment and the size must be coprime to allow using all of the queue slots.");

  static constexpr auto kMemOrder = cuda::std::memory_order_relaxed;

  explicit resource_queue_t(uint32_t capacity = Size) noexcept : capacity_{capacity}
  {
    head_.store(0, kMemOrder);
    tail_.store(0, kMemOrder);
    for (uint32_t i = 0; i < kSize; i++) {
      buf_[i].store(kEmpty, kMemOrder);
    }
  }

  /** Nominal capacity of the queue. */
  [[nodiscard]] auto capacity() const { return capacity_; }

  /** This does not affect the queue behavior, but merely declares a nominal capacity. */
  void set_capacity(uint32_t capacity) { capacity_ = capacity; }

  /**
   * A slot in the queue to take the value from.
   * Once it's obtained, the corresponding value in the queue is lost for other users.
   */
  struct promise_t {
    explicit promise_t(cuda::std::atomic<value_type>& loc) : loc_{loc}, val_{Empty} {}
    ~promise_t() noexcept { wait(); }

    auto test() noexcept -> bool
    {
      if (val_ != Empty) { return true; }
      val_ = loc_.exchange(kEmpty, kMemOrder);
      return val_ != Empty;
    }

    auto test(value_type& e) noexcept -> bool
    {
      if (test()) {
        e = val_;
        return true;
      }
      return false;
    }

    auto wait() noexcept -> value_type
    {
      if (val_ == Empty) {
        // [HOT SPOT]
        // Optimize for the case of contention: expect the loc is empty.
        do {
          loc_.wait(kEmpty, kMemOrder);
          val_ = loc_.exchange(kEmpty, kMemOrder);
        } while (val_ == kEmpty);
      }
      return val_;
    }

   private:
    cuda::std::atomic<value_type>& loc_;
    value_type val_;
  };

  void push(value_type x) noexcept
  {
    auto& loc = buf_[head_.fetch_add(kCounterIncrement, kMemOrder) & kCounterLocMask];
    /* [NOT A HOT SPOT]
     We expect there's always enough place in the queue to push the item,
     but also we expect a few pop waiters - notify them the data is available.
     */
    value_type e = kEmpty;
    while (!loc.compare_exchange_weak(e, x, kMemOrder, kMemOrder)) {
      e = kEmpty;
    }
    loc.notify_one();
  }

  auto pop() noexcept -> promise_t
  {
    auto& loc = buf_[tail_.fetch_add(kCounterIncrement, kMemOrder) & kCounterLocMask];
    return promise_t{loc};
  }

 private:
  alignas(kCacheLineBytes) cuda::std::atomic<uint32_t> head_{};
  alignas(kCacheLineBytes) cuda::std::atomic<uint32_t> tail_{};
  alignas(kCacheLineBytes) std::array<cuda::std::atomic<value_type>, kSize> buf_{};
  alignas(kCacheLineBytes) uint32_t capacity_;
};

/** Primitive fixed-size deque for single-threaded use. */
template <typename T>
struct local_deque_t {
  explicit local_deque_t(uint32_t size) : store_(size) {}

  [[nodiscard]] auto capacity() const -> uint32_t { return store_.size(); }
  [[nodiscard]] auto size() const -> uint32_t { return end_ - start_; }

  void push_back(T x) { store_[end_++ % capacity()] = x; }

  void push_front(T x)
  {
    if (start_ == 0) {
      start_ += capacity();
      end_ += capacity();
    }
    store_[--start_ % capacity()] = x;
  }

  // NB: unsafe functions - do not check if the queue is full/empty.
  auto pop_back() -> T { return store_[--end_ % capacity()]; }
  auto pop_front() -> T { return store_[start_++ % capacity()]; }

  auto try_push_back(T x) -> bool
  {
    if (size() >= capacity()) { return false; }
    push_back(x);
    return true;
  }

  auto try_push_front(T x) -> bool
  {
    if (size() >= capacity()) { return false; }
    push_front(x);
    return true;
  }

  auto try_pop_back(T& x) -> bool
  {
    if (start_ >= end_) { return false; }
    x = pop_back();
    return true;
  }

  auto try_pop_front(T& x) -> bool
  {
    if (start_ >= end_) { return false; }
    x = pop_front();
    return true;
  }

 private:
  std::vector<T> store_;
  uint32_t start_{0};
  uint32_t end_{0};
};

struct persistent_runner_base_t {
  using job_queue_type    = resource_queue_t<uint32_t, kMaxJobsNum>;
  using worker_queue_type = resource_queue_t<uint32_t, kMaxWorkersNum>;
  rmm::mr::pinned_host_memory_resource worker_handles_mr;
  rmm::mr::pinned_host_memory_resource job_descriptor_mr;
  rmm::mr::cuda_memory_resource device_mr;
  cudaStream_t stream{};
  job_queue_type job_queue{};
  worker_queue_type worker_queue{};
  // This should be large enough to make the runner live through restarts of the benchmark cases.
  // Otherwise, the benchmarks slowdown significantly.
  std::chrono::milliseconds lifetime;

  persistent_runner_base_t(float persistent_lifetime)
    : lifetime(size_t(persistent_lifetime * 1000)), job_queue(), worker_queue()
  {
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  virtual ~persistent_runner_base_t() noexcept { cudaStreamDestroy(stream); };
};

struct alignas(kCacheLineBytes) launcher_t {
  using job_queue_type           = persistent_runner_base_t::job_queue_type;
  using worker_queue_type        = persistent_runner_base_t::worker_queue_type;
  using pending_reads_queue_type = local_deque_t<uint32_t>;
  using completion_flag_type     = cuda::atomic<bool, cuda::thread_scope_system>;

  pending_reads_queue_type pending_reads;
  job_queue_type& job_ids;
  worker_queue_type& idle_worker_ids;
  worker_handle_t* worker_handles;
  uint32_t job_id;
  completion_flag_type* completion_flag;
  bool all_done = false;

  /* [Note: sleeping]
  When the number of threads is greater than the number of cores, the threads start to fight for
  the CPU time, which reduces the throughput.
  To ease the competition, we track the expected GPU latency and let a thread sleep for some
  time, and only start to spin when it's about a time to get the result.
  */
  static inline constexpr auto kDefaultLatency = std::chrono::nanoseconds(50000);
  /* This is the base for computing maximum time a thread is allowed to sleep. */
  static inline constexpr auto kMaxExpectedLatency =
    kDefaultLatency * std::max<std::uint32_t>(10, kMaxJobsNum / 128);
  static inline thread_local auto expected_latency = kDefaultLatency;
  const std::chrono::time_point<std::chrono::system_clock> start;
  std::chrono::time_point<std::chrono::system_clock> now;
  const int64_t pause_factor;
  int pause_count = 0;
  /**
   * Beyond this threshold, the launcher (calling thread) does not wait for the results anymore and
   * throws an exception.
   */
  std::chrono::time_point<std::chrono::system_clock> deadline;

  template <typename RecordWork>
  launcher_t(job_queue_type& job_ids,
             worker_queue_type& idle_worker_ids,
             worker_handle_t* worker_handles,
             uint32_t n_queries,
             std::chrono::milliseconds max_wait_time,
             RecordWork record_work)
    : pending_reads{std::min(n_queries, kMaxWorkersPerThread)},
      job_ids{job_ids},
      idle_worker_ids{idle_worker_ids},
      worker_handles{worker_handles},
      job_id{job_ids.pop().wait()},
      completion_flag{record_work(job_id)},
      start{std::chrono::system_clock::now()},
      pause_factor{calc_pause_factor(n_queries)},
      now{start},
      deadline{start + max_wait_time + expected_latency}
  {
    // Wait for the first worker and submit the query immediately.
    submit_query(idle_worker_ids.pop().wait(), 0);
    // Submit the rest of the queries in the batch
    for (uint32_t i = 1; i < n_queries; i++) {
      auto promised_worker = idle_worker_ids.pop();
      uint32_t worker_id;
      while (!promised_worker.test(worker_id)) {
        if (pending_reads.try_pop_front(worker_id)) {
          bool returned_some = false;
          for (bool keep_returning = true; keep_returning;) {
            if (try_return_worker(worker_id)) {
              keep_returning = pending_reads.try_pop_front(worker_id);
              returned_some  = true;
            } else {
              pending_reads.push_front(worker_id);
              keep_returning = false;
            }
          }
          if (!returned_some) { pause(); }
        } else {
          // Calmly wait for the promised worker instead of spinning.
          worker_id = promised_worker.wait();
          break;
        }
      }
      pause_count = 0;  // reset the pause behavior
      submit_query(worker_id, i);
      // Try to not hold too many workers in one thread
      if (i >= kSoftMaxWorkersPerThread && pending_reads.try_pop_front(worker_id)) {
        if (!try_return_worker(worker_id)) { pending_reads.push_front(worker_id); }
      }
    }
  }

  inline ~launcher_t() noexcept  // NOLINT
  {
    // bookkeeping: update the expected latency to wait more efficiently later
    constexpr size_t kWindow = 100;  // moving average memory
    expected_latency         = std::min<std::chrono::nanoseconds>(
      ((kWindow - 1) * expected_latency + now - start) / kWindow, kMaxExpectedLatency);

    // Try to gracefully cleanup the queue resources if the launcher is being destructed after an
    // exception.
    if (job_id != job_queue_type::kEmpty) { job_ids.push(job_id); }
    uint32_t worker_id;
    while (pending_reads.try_pop_front(worker_id)) {
      idle_worker_ids.push(worker_id);
    }
  }

  inline void submit_query(uint32_t worker_id, uint32_t query_id)
  {
    worker_handles[worker_id].data.store(worker_handle_t::data_t{.value = {job_id, query_id}},
                                         cuda::memory_order_relaxed);

    while (!pending_reads.try_push_back(worker_id)) {
      // The only reason pending_reads cannot push is that the queue is full.
      // It's local, so we must pop and wait for the returned worker to finish its work.
      auto pending_worker_id = pending_reads.pop_front();
      while (!try_return_worker(pending_worker_id)) {
        pause();
      }
    }
    pause_count = 0;  // reset the pause behavior
  }

  /** Check if the worker has finished the work; if so, return it to the shared pool. */
  inline auto try_return_worker(uint32_t worker_id) -> bool
  {
    // Use the cached `all_done` - makes sense when called from the `wait()` routine.
    if (all_done ||
        !is_worker_busy(worker_handles[worker_id].data.load(cuda::memory_order_relaxed).handle)) {
      idle_worker_ids.push(worker_id);
      return true;
    } else {
      return false;
    }
  }

  /** Check if all workers finished their work. */
  inline auto is_all_done()
  {
    // Cache the result of the check to avoid doing unnecessary atomic loads.
    if (all_done) { return true; }
    all_done = completion_flag->load(cuda::memory_order_relaxed);
    return all_done;
  }

  /** The launcher shouldn't attempt to wait past the returned time. */
  [[nodiscard]] inline auto sleep_limit() const
  {
    constexpr auto kMinWakeTime  = std::chrono::nanoseconds(10000);
    constexpr double kSleepLimit = 0.6;
    return start + expected_latency * kSleepLimit - kMinWakeTime;
  }

  /**
   * When the latency is much larger than expected, it's a sign that there is a thread contention.
   * Then we switch to sleeping instead of waiting to give the cpu cycles to other threads.
   */
  [[nodiscard]] inline auto overtime_threshold() const
  {
    constexpr auto kOvertimeFactor = 3;
    return start + expected_latency * kOvertimeFactor;
  }

  /**
   * Calculate the fraction of time can be spent sleeping in a single call to `pause()`.
   * Naturally it depends on the number of queries in a batch and the number of parallel workers.
   */
  [[nodiscard]] inline auto calc_pause_factor(uint32_t n_queries) const -> uint32_t
  {
    constexpr uint32_t kMultiplier = 10;
    return kMultiplier * raft::div_rounding_up_safe(n_queries, idle_worker_ids.capacity());
  }

  /** Wait a little bit (called in a loop). */
  inline void pause()
  {
    // Don't sleep this many times hoping for smoother run
    constexpr auto kSpinLimit = 3;
    // It doesn't make much sense to slee less than this
    constexpr auto kPauseTimeMin = std::chrono::nanoseconds(1000);
    // Bound sleeping time
    constexpr auto kPauseTimeMax = std::chrono::nanoseconds(50000);
    if (pause_count++ < kSpinLimit) {
      std::this_thread::yield();
      return;
    }
    now                  = std::chrono::system_clock::now();
    auto pause_time_base = std::max(now - start, expected_latency);
    auto pause_time      = std::clamp(pause_time_base / pause_factor, kPauseTimeMin, kPauseTimeMax);
    if (now + pause_time < sleep_limit()) {
      // It's too early: sleep for a bit
      std::this_thread::sleep_for(pause_time);
    } else if (now <= overtime_threshold()) {
      // It's about time to check the results, don't sleep
      std::this_thread::yield();
    } else if (now <= deadline) {
      // Too late; perhaps the system is too busy - sleep again
      std::this_thread::sleep_for(pause_time);
    } else {
      // Missed the deadline: throw an exception
      throw raft::exception(
        "The calling thread didn't receive the results from the persistent CAGRA kernel within the "
        "expected kernel lifetime. Here are possible reasons of this failure:\n"
        "  (1) `persistent_lifetime` search parameter is too small - increase it;\n"
        "  (2) there is other work being executed on the same device and the kernel failed to "
        "progress - decreasing `persistent_device_usage` may help (but not guaranteed);\n"
        "  (3) there is a bug in the implementation - please report it to cuVS team.");
    }
  }

  /** Wait for all work to finish and don't forget to return the workers to the shared pool. */
  inline void wait()
  {
    uint32_t worker_id;
    while (pending_reads.try_pop_front(worker_id)) {
      while (!try_return_worker(worker_id)) {
        if (!is_all_done()) { pause(); }
      }
    }
    pause_count = 0;  // reset the pause behavior
    // terminal state, should be engaged only after the `pending_reads` is empty
    // and `queries_submitted == n_queries`
    now = std::chrono::system_clock::now();
    while (!is_all_done()) {
      auto till_time = sleep_limit();
      if (now < till_time) {
        std::this_thread::sleep_until(till_time);
        now = std::chrono::system_clock::now();
      } else {
        pause();
      }
    }

    // Return the job descriptor
    job_ids.push(job_id);
    job_id = job_queue_type::kEmpty;
  }
};

template <typename DataT, typename IndexT, typename DistanceT, typename SampleFilterT>
struct alignas(kCacheLineBytes) persistent_runner_t : public persistent_runner_base_t {
  using descriptor_base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using index_type           = IndexT;
  using distance_type        = DistanceT;
  using data_type            = DataT;
  using kernel_config_type   = search_kernel_config<true, descriptor_base_type, SampleFilterT>;
  using kernel_type          = typename kernel_config_type::kernel_t;
  using job_desc_type        = job_desc_t<descriptor_base_type>;
  kernel_type kernel;
  uint32_t block_size;
  dataset_descriptor_host<DataT, IndexT, DistanceT> dd_host;
  rmm::device_uvector<worker_handle_t> worker_handles;
  rmm::device_uvector<job_desc_type> job_descriptors;
  rmm::device_uvector<uint32_t> completion_counters;
  rmm::device_uvector<index_type> hashmap;
  std::atomic<std::chrono::time_point<std::chrono::system_clock>> last_touch;
  uint64_t param_hash;

  /**
   * Calculate the hash of the parameters to detect if they've changed across the calls.
   * NB: this must have the same argument types as the constructor.
   */
  static inline auto calculate_parameter_hash(
    std::reference_wrapper<const dataset_descriptor_host<DataT, IndexT, DistanceT>> dataset_desc,
    raft::device_matrix_view<const index_type, int64_t, raft::row_major> graph,
    uint32_t num_itopk_candidates,
    uint32_t block_size,  //
    uint32_t smem_size,
    int64_t hash_bitlen,
    size_t small_hash_bitlen,
    size_t small_hash_reset_interval,
    uint32_t num_random_samplings,
    uint64_t rand_xor_mask,
    uint32_t num_seeds,
    size_t itopk_size,
    size_t search_width,
    size_t min_iterations,
    size_t max_iterations,
    SampleFilterT sample_filter,
    float persistent_lifetime,
    float persistent_device_usage) -> uint64_t
  {
    return uint64_t(graph.data_handle()) ^ dataset_desc.get().team_size ^ num_itopk_candidates ^
           block_size ^ smem_size ^ hash_bitlen ^ small_hash_reset_interval ^ num_random_samplings ^
           rand_xor_mask ^ num_seeds ^ itopk_size ^ search_width ^ min_iterations ^ max_iterations ^
           uint64_t(persistent_lifetime * 1000) ^ uint64_t(persistent_device_usage * 1000);
  }

  persistent_runner_t(
    std::reference_wrapper<const dataset_descriptor_host<DataT, IndexT, DistanceT>> dataset_desc,
    raft::device_matrix_view<const index_type, int64_t, raft::row_major> graph,
    uint32_t num_itopk_candidates,
    uint32_t block_size,  //
    uint32_t smem_size,
    int64_t hash_bitlen,
    size_t small_hash_bitlen,
    size_t small_hash_reset_interval,
    uint32_t num_random_samplings,
    uint64_t rand_xor_mask,
    uint32_t num_seeds,
    size_t itopk_size,
    size_t search_width,
    size_t min_iterations,
    size_t max_iterations,
    SampleFilterT sample_filter,
    float persistent_lifetime,
    float persistent_device_usage)
    : persistent_runner_base_t{persistent_lifetime},
      kernel{kernel_config_type::choose_itopk_and_mx_candidates(
        itopk_size, num_itopk_candidates, block_size)},
      block_size{block_size},
      worker_handles(0, stream, worker_handles_mr),
      job_descriptors(kMaxJobsNum, stream, job_descriptor_mr),
      completion_counters(kMaxJobsNum, stream, device_mr),
      hashmap(0, stream, device_mr),
      dd_host{dataset_desc.get()},
      param_hash(calculate_parameter_hash(dd_host,
                                          graph,
                                          num_itopk_candidates,
                                          block_size,
                                          smem_size,
                                          hash_bitlen,
                                          small_hash_bitlen,
                                          small_hash_reset_interval,
                                          num_random_samplings,
                                          rand_xor_mask,
                                          num_seeds,
                                          itopk_size,
                                          search_width,
                                          min_iterations,
                                          max_iterations,
                                          sample_filter,
                                          persistent_lifetime,
                                          persistent_device_usage))
  {
    // initialize the dataset/distance descriptor
    auto* dd_dev_ptr = dd_host.dev_ptr(stream);

    // set kernel attributes same as in normal kernel
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    // set kernel launch parameters
    dim3 gs = calc_coop_grid_size(block_size, smem_size, persistent_device_usage);
    dim3 bs(block_size, 1, 1);
    RAFT_LOG_DEBUG(
      "Launching persistent kernel with %u threads, %u block %u smem", bs.x, gs.y, smem_size);

    // initialize the job queue
    auto* completion_counters_ptr = completion_counters.data();
    auto* job_descriptors_ptr     = job_descriptors.data();
    for (uint32_t i = 0; i < kMaxJobsNum; i++) {
      auto& jd                = job_descriptors_ptr[i].input.value;
      jd.result_indices_ptr   = 0;
      jd.result_distances_ptr = nullptr;
      jd.queries_ptr          = nullptr;
      jd.top_k                = 0;
      jd.n_queries            = 0;
      job_descriptors_ptr[i].completion_flag.store(false);
      job_queue.push(i);
    }

    // initialize the worker queue
    worker_queue.set_capacity(gs.y);
    worker_handles.resize(gs.y, stream);
    auto* worker_handles_ptr = worker_handles.data();
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    for (uint32_t i = 0; i < gs.y; i++) {
      worker_handles_ptr[i].data.store({kWaitForWork});
      worker_queue.push(i);
    }

    index_type* hashmap_ptr = nullptr;
    if (small_hash_bitlen == 0) {
      hashmap.resize(gs.y * hashmap::get_size(hash_bitlen), stream);
      hashmap_ptr = hashmap.data();
    }

    // launch the kernel
    auto* graph_ptr                   = graph.data_handle();
    uint32_t graph_degree             = graph.extent(1);
    uint32_t* num_executed_iterations = nullptr;  // optional arg [num_queries]
    const index_type* dev_seed_ptr    = nullptr;  // optional arg [num_queries, num_seeds]

    void* args[] =  // NOLINT
      {&dd_dev_ptr,
       &worker_handles_ptr,
       &job_descriptors_ptr,
       &completion_counters_ptr,
       &graph_ptr,  // [dataset_size, graph_degree]
       &graph_degree,
       &num_random_samplings,
       &rand_xor_mask,
       &dev_seed_ptr,
       &num_seeds,
       &hashmap_ptr,  // visited_hashmap_ptr: [num_queries, 1 << hash_bitlen]
       &itopk_size,
       &search_width,
       &min_iterations,
       &max_iterations,
       &num_executed_iterations,
       &hash_bitlen,
       &small_hash_bitlen,
       &small_hash_reset_interval,
       &sample_filter};
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    RAFT_CUDA_TRY(cudaLaunchCooperativeKernel<std::remove_pointer_t<kernel_type>>(
      kernel, gs, bs, args, smem_size, stream));
    RAFT_LOG_INFO(
      "Initialized the kernel %p in stream %zd; job_queue size = %u; worker_queue size = %u",
      reinterpret_cast<void*>(kernel),
      int64_t((cudaStream_t)stream),
      job_queue.capacity(),
      worker_queue.capacity());
    last_touch.store(std::chrono::system_clock::now(), std::memory_order_relaxed);
  }

  ~persistent_runner_t() noexcept override
  {
    auto whs = worker_handles.data();
    for (auto i = worker_handles.size(); i > 0; i--) {
      whs[worker_queue.pop().wait()].data.store({kNoMoreWork}, cuda::memory_order_relaxed);
    }
    RAFT_CUDA_TRY_NO_THROW(cudaStreamSynchronize(stream));
    RAFT_LOG_INFO("Destroyed the persistent runner.");
  }

  void launch(uintptr_t result_indices_ptr,         // [num_queries, top_k]
              distance_type* result_distances_ptr,  // [num_queries, top_k]
              const data_type* queries_ptr,         // [num_queries, dataset_dim]
              uint32_t num_queries,
              uint32_t top_k)
  {
    // submit all queries
    launcher_t launcher{job_queue,
                        worker_queue,
                        worker_handles.data(),
                        num_queries,
                        this->lifetime,
                        [=](uint32_t job_ix) {
                          auto& jd                = job_descriptors.data()[job_ix].input.value;
                          auto* cflag             = &job_descriptors.data()[job_ix].completion_flag;
                          jd.result_indices_ptr   = result_indices_ptr;
                          jd.result_distances_ptr = result_distances_ptr;
                          jd.queries_ptr          = queries_ptr;
                          jd.top_k                = top_k;
                          jd.n_queries            = num_queries;
                          cflag->store(false, cuda::memory_order_relaxed);
                          cuda::atomic_thread_fence(cuda::memory_order_release,
                                                    cuda::thread_scope_system);
                          return cflag;
                        }};

    // Update the state of the keep-alive atomic in the meanwhile
    auto prev_touch = last_touch.load(std::memory_order_relaxed);
    if (prev_touch + lifetime / 10 < launcher.now) {
      // to avoid congestion at this atomic, we only update it if a significant fraction of the live
      // interval has passed.
      last_touch.store(launcher.now, std::memory_order_relaxed);
    }
    // wait for the results to arrive
    launcher.wait();
  }

  auto calc_coop_grid_size(uint32_t block_size, uint32_t smem_size, float persistent_device_usage)
    -> dim3
  {
    // determine the grid size
    int ctas_per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor<kernel_type>(
      &ctas_per_sm, kernel, block_size, smem_size);
    int num_sm    = raft::getMultiProcessorCount();
    auto n_blocks = static_cast<uint32_t>(persistent_device_usage * (ctas_per_sm * num_sm));
    if (n_blocks > kMaxWorkersNum) {
      RAFT_LOG_WARN("Limiting the grid size limit due to the size of the queue: %u -> %u",
                    n_blocks,
                    kMaxWorkersNum);
      n_blocks = kMaxWorkersNum;
    }

    return {1, n_blocks, 1};
  }
};

struct alignas(kCacheLineBytes) persistent_state {
  std::shared_ptr<persistent_runner_base_t> runner{nullptr};
  std::mutex lock;
};

inline persistent_state persistent{};

template <typename RunnerT, typename... Args>
auto create_runner(Args... args) -> std::shared_ptr<RunnerT>  // it's ok.. pass everything by values
{
  std::lock_guard<std::mutex> guard(persistent.lock);
  // Check if the runner has already been created
  std::shared_ptr<RunnerT> runner_outer = std::dynamic_pointer_cast<RunnerT>(persistent.runner);
  if (runner_outer) {
    if (runner_outer->param_hash == RunnerT::calculate_parameter_hash(args...)) {
      return runner_outer;
    } else {
      runner_outer.reset();
    }
  }
  // Runner has not yet been created (or it's incompatible):
  //   create it in another thread and only then release the lock.
  // Free the resources (if any) in advance
  persistent.runner.reset();

  cuda::std::atomic_flag ready{};
  ready.clear(cuda::std::memory_order_relaxed);
  std::thread(
    [&runner_outer, &ready](Args... thread_args) {  // pass everything by values
      // create the runner (the lock is acquired in the parent thread).
      runner_outer      = std::make_shared<RunnerT>(thread_args...);
      auto lifetime     = runner_outer->lifetime;
      persistent.runner = std::static_pointer_cast<persistent_runner_base_t>(runner_outer);
      std::weak_ptr<RunnerT> runner_weak = runner_outer;
      ready.test_and_set(cuda::std::memory_order_release);
      ready.notify_one();
      // NB: runner_outer is passed by reference and may be dead by this time.

      while (true) {
        std::this_thread::sleep_for(lifetime);
        auto runner = runner_weak.lock();  // runner_weak is local - thread-safe
        if (!runner) {
          return;  // dead already
        }
        if (runner->last_touch.load(std::memory_order_relaxed) + lifetime <
            std::chrono::system_clock::now()) {
          std::lock_guard<std::mutex> guard(persistent.lock);
          if (runner == persistent.runner) { persistent.runner.reset(); }
          return;
        }
      }
    },
    args...)
    .detach();
  ready.wait(false, cuda::std::memory_order_acquire);
  return runner_outer;
}

template <typename RunnerT, typename... Args>
auto get_runner(Args... args) -> std::shared_ptr<RunnerT>
{
  // Using a thread-local weak pointer allows us to avoid using locks/atomics,
  // since the control block of weak/shared pointers is thread-safe.
  static thread_local std::weak_ptr<RunnerT> weak;
  auto runner = weak.lock();
  if (runner) {
    if (runner->param_hash == RunnerT::calculate_parameter_hash(args...)) {
      return runner;
    } else {
      weak.reset();
      runner.reset();
    }
  }
  // Thread-local variable expected_latency makes sense only for a current RunnerT configuration.
  // If `weak` is not alive, it's a hint the configuration has changed and we should reset our
  // estimate of the expected launch latency.
  launcher_t::expected_latency = launcher_t::kDefaultLatency;
  runner                       = create_runner<RunnerT>(args...);
  weak                         = runner;
  return runner;
}

template <typename DataT, typename IndexT, typename DistanceT, typename SampleFilterT>
void select_and_run(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                    raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
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
  if (ps.persistent) {
    using runner_type = persistent_runner_t<DataT, IndexT, DistanceT, SampleFilterT>;

    get_runner<runner_type>(/*
Note, we're passing the descriptor by reference here, and this reference is going to be passed to a
new spawned thread, which is dangerous. However, the descriptor is copied in that thread before the
control is returned in this thread (in persistent_runner_t constructor), so we're safe.
*/
                            std::cref(dataset_desc),
                            graph,
                            num_itopk_candidates,
                            block_size,
                            smem_size,
                            hash_bitlen,
                            small_hash_bitlen,
                            small_hash_reset_interval,
                            ps.num_random_samplings,
                            ps.rand_xor_mask,
                            num_seeds,
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
    auto kernel                = search_kernel_config<false, descriptor_base_type, SampleFilterT>::
      choose_itopk_and_mx_candidates(ps.itopk_size, num_itopk_candidates, block_size);
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 thread_dims(block_size, 1, 1);
    dim3 block_dims(1, num_queries, 1);
    RAFT_LOG_DEBUG(
      "Launching kernel with %u threads, %u block %u smem", block_size, num_queries, smem_size);
    kernel<<<block_dims, thread_dims, smem_size, stream>>>(topk_indices_ptr,
                                                           topk_distances_ptr,
                                                           topk,
                                                           dataset_desc.dev_ptr(stream),
                                                           queries_ptr,
                                                           graph.data_handle(),
                                                           graph.extent(1),
                                                           ps.num_random_samplings,
                                                           ps.rand_xor_mask,
                                                           dev_seed_ptr,
                                                           num_seeds,
                                                           hashmap_ptr,
                                                           ps.itopk_size,
                                                           ps.search_width,
                                                           ps.min_iterations,
                                                           ps.max_iterations,
                                                           num_executed_iterations,
                                                           hash_bitlen,
                                                           small_hash_bitlen,
                                                           small_hash_reset_interval,
                                                           sample_filter);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}
}  // namespace single_cta_search
}  // namespace cuvs::neighbors::cagra::detail
