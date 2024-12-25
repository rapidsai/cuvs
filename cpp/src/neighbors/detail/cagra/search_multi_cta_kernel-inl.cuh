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

#include "search_multi_cta_kernel.cuh"

#include "bitonic.hpp"
#include "compute_distance-ext.cuh"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_for_cagra/topk.h"  // TODO replace with raft topk if possible
#include "utils.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger-ext.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/distance/distance.hpp>

#include <cuvs/neighbors/common.hpp>

// TODO: This shouldn't be invoking anything from spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace cuvs::neighbors::cagra::detail {
namespace multi_cta_search {

// #define _CLK_BREAKDOWN

template <class INDEX_T, class DISTANCE_T>
RAFT_DEVICE_INLINE_FUNCTION void pickup_next_parents(
  INDEX_T* const next_parent_indices,  // [num_parents]
  const uint32_t num_parents,
  INDEX_T* const itopk_indices,       // [num_itopk]
  DISTANCE_T* const itopk_distances,  // [num_itopk]
  const uint32_t num_itopk,           // (*) num_itopk <= 32
  INDEX_T* const hash_ptr,
  const uint32_t hash_bitlen)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);

  const unsigned warp_id = threadIdx.x / 32;
  if (warp_id > 0) { return; }
  const unsigned lane_id = threadIdx.x % 32;

  // Initialize
  if (lane_id < num_parents) { next_parent_indices[lane_id] = ~static_cast<INDEX_T>(0); }
  INDEX_T index = invalid_index;
  if (lane_id < num_itopk) { index = itopk_indices[lane_id]; }

  int is_candidate = 0;
  if ((index != invalid_index) && ((index & index_msb_1_mask) == 0)) {
#if 0
    if (hashmap::search<INDEX_T, 1>(hash_ptr, hash_bitlen, index)) {
      // Deactivate nodes that have already been used by other CTAs.
      itopk_indices[lane_id]   = invalid_index;
      itopk_distances[lane_id] = utils::get_max_value<DISTANCE_T>();
      index                    = invalid_index;
    } else {
      is_candidate = 1;
    }
#else
    is_candidate = 1;
#endif
  }

  uint32_t num_next_parents = 0;
  while (num_next_parents < num_parents) {
    const uint32_t ballot_mask = __ballot_sync(0xffffffff, is_candidate);
    int num_candidates         = __popc(ballot_mask);
    if (num_candidates == 0) { return; }
    int is_found = 0;
    if (is_candidate) {
      const auto candidate_id = __popc(ballot_mask & ((1 << lane_id) - 1));
      if (candidate_id == 0) {
        if (hashmap::insert<INDEX_T, 1>(hash_ptr, hash_bitlen, index)) {
          // Use this candidate as next parent
          next_parent_indices[num_next_parents] = lane_id;
          index |= index_msb_1_mask;  // set most significant bit as used node
          is_found = 1;
        } else {
          // Deactivate the node since it has been used by other CTA.
          index                    = invalid_index;
          itopk_distances[lane_id] = utils::get_max_value<DISTANCE_T>();
        }
        itopk_indices[lane_id] = index;
        is_candidate           = 0;
      }
    }
    if (__ballot_sync(0xffffffff, is_found)) { num_next_parents += 1; }
  }
}

template <unsigned MAX_ELEMENTS, class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void topk_by_bitonic_sort(float* distances,  // [num_elements]
                                                      INDEX_T* indices,  // [num_elements]
                                                      const uint32_t num_elements)
{
  const unsigned warp_id = threadIdx.x / 32;
  if (warp_id > 0) { return; }
  const unsigned lane_id = threadIdx.x % 32;
  constexpr unsigned N   = (MAX_ELEMENTS + 31) / 32;
  float key[N];
  INDEX_T val[N];
  for (unsigned i = 0; i < N; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < num_elements) {
      key[i] = distances[j];
      val[i] = indices[j];
    } else {
      key[i] = utils::get_max_value<float>();
      val[i] = ~static_cast<INDEX_T>(0);
    }
  }
  /* Warp Sort */
  bitonic::warp_sort<float, INDEX_T, N>(key, val);
  /* Store sorted results */
  for (unsigned i = 0; i < N; i++) {
    unsigned j = (N * lane_id) + i;
    if (j < num_elements) {
      distances[j] = key[i];
      indices[j]   = val[i];
    }
  }
}

template <class INDEX_T, class DISTANCE_T>
RAFT_DEVICE_INLINE_FUNCTION void move_valid_entries_to_head(
  INDEX_T* indices,            // [num_elements]
  DISTANCE_T* distances,       // [num_elements]
  const uint32_t num_elements  // (*) num_elements must be multiple of 32
)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  constexpr INDEX_T invalid_index    = ~static_cast<INDEX_T>(0);
  const uint32_t warp_id             = threadIdx.x / 32;
  if (warp_id > 0) { return; }
  const uint32_t lane_id = threadIdx.x % 32;
  uint32_t offset        = 0;
  for (uint32_t i = lane_id; i < num_elements; i += 32) {
    auto index      = indices[i];
    auto distance   = distances[i];
    bool is_valid   = (index != invalid_index);
    const auto mask = __ballot_sync(0xffffffff, is_valid);
    const auto j    = offset + __popc(mask & ((1 << lane_id) - 1));
    if ((index != invalid_index) && (j < i)) {
      indices[j]   = index;
      distances[j] = distance;
    }
    offset += __popc(mask);
    __syncwarp();
  }
  for (uint32_t i = offset + lane_id; i < num_elements; i += 32) {
    indices[i]   = invalid_index;
    distances[i] = utils::get_max_value<DISTANCE_T>();
  }
  __syncwarp();
}

//
// multiple CTAs per single query
//
template <std::uint32_t MAX_ELEMENTS, class DATASET_DESCRIPTOR_T, class SAMPLE_FILTER_T>
RAFT_KERNEL __launch_bounds__(1024, 1) search_kernel(
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    result_indices_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const
    result_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  const DATASET_DESCRIPTOR_T* dataset_desc,
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const typename DATASET_DESCRIPTOR_T::INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  const uint32_t visited_hash_bitlen,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const
    traversed_hashmap_ptr,  // [num_queries, 1 << traversed_hash_bitlen]
  const uint32_t traversed_hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t search_width,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations, /* stats */
  SAMPLE_FILTER_T sample_filter)
{
  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;

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
  // +----------------+---------+-------------------------------+
  // | internal_top_k | padding | neighbors of parent nodes     |
  // | <itopk_size>   | upto 32 | <search_width * graph_degree> |
  // +----------------+---------+-------------------------------+
  // |<---        result_buffer_size_32                     --->|
  const auto result_buffer_size    = itopk_size + (search_width * graph_degree);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  assert(result_buffer_size_32 <= MAX_ELEMENTS);

  // Set smem working buffer for the distance calculation
  dataset_desc = dataset_desc->setup_workspace(smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<INDEX_T*>(smem + dataset_desc->smem_ws_size_in_bytes());
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ local_visited_hashmap_ptr =
    reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_indices_buffer =
    reinterpret_cast<INDEX_T*>(local_visited_hashmap_ptr + hashmap::get_size(visited_hash_bitlen));

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

  // compute distance to randomly selecting nodes
  _CLK_START();
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  uint32_t block_id                   = cta_id + (num_cta_per_query * query_id);
  uint32_t num_blocks                 = num_cta_per_query * num_queries;

  device::compute_distance_to_random_nodes(result_indices_buffer,
                                           result_distances_buffer,
                                           *dataset_desc,
                                           graph_degree * search_width,
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
    // Checks the state of the node in the result buffer from the previous
    // iteration, and if it cannot be used as a parent node, it is deactivated.
    for (uint32_t i = threadIdx.x; i < result_buffer_size_32 - (graph_degree * search_width);
         i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index || index & index_msb_1_mask) { continue; }
      if (hashmap::search<INDEX_T, 1>(local_traversed_hashmap_ptr, traversed_hash_bitlen, index)) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
    __syncthreads();

    // Topk with bitonic sort (1st warp only)
    topk_by_bitonic_sort<MAX_ELEMENTS, INDEX_T>(
      result_distances_buffer, result_indices_buffer, result_buffer_size_32);
    __syncthreads();
    _CLK_REC(clk_topk);

    if (iter + 1 >= max_iteration) { break; }

    _CLK_START();
    if (threadIdx.x < 32) {
      // [1st warp] Pick up next parents
      pickup_next_parents<INDEX_T, DISTANCE_T>(parent_indices_buffer,
                                               search_width,
                                               result_indices_buffer,
                                               result_distances_buffer,
                                               itopk_size,
                                               local_traversed_hashmap_ptr,
                                               traversed_hash_bitlen);
#if 1
      if (parent_indices_buffer[0] == invalid_index) {
        // Try again if no parent is found
        move_valid_entries_to_head<INDEX_T, DISTANCE_T>(
          result_indices_buffer, result_distances_buffer, result_buffer_size_32);
        pickup_next_parents<INDEX_T, DISTANCE_T>(parent_indices_buffer,
                                                 search_width,
                                                 result_indices_buffer,
                                                 result_distances_buffer,
                                                 itopk_size,
                                                 local_traversed_hashmap_ptr,
                                                 traversed_hash_bitlen);
      }
#endif
    } else {
      // [Other warps] Reset visited hashmap
      hashmap::init<INDEX_T>(local_visited_hashmap_ptr, visited_hash_bitlen, 32);
    }
    __syncthreads();
    _CLK_REC(clk_pickup_parents);

    if ((parent_indices_buffer[0] == invalid_index) && (iter >= min_iteration)) { break; }

    if (threadIdx.x < 32) {
      // [1st warp] Restore visited hashmap by putting itopk indices in it.
      for (unsigned i = threadIdx.x; i < itopk_size; i += 32) {
        INDEX_T index = result_indices_buffer[i];
        if (index == invalid_index) { continue; }
        index &= ~index_msb_1_mask;
        hashmap::insert(local_visited_hashmap_ptr, visited_hash_bitlen, index);
      }
    } else {
      // [Other warps] Remove entries kicked out of the itopk list from the
      // traversed hash table.
      for (unsigned i = itopk_size + threadIdx.x - 32; i < result_buffer_size_32;
           i += blockDim.x - 32) {
        INDEX_T index = result_indices_buffer[i];
        if (index == invalid_index) { continue; }
        if (index & index_msb_1_mask) {
          hashmap::remove<INDEX_T>(
            local_traversed_hashmap_ptr, traversed_hash_bitlen, index & ~index_msb_1_mask);
          result_indices_buffer[i]   = invalid_index;
          result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
        }
      }
    }
    __syncthreads();

    _CLK_START();
    // compute the norms between child nodes and query node
    device::compute_distance_to_child_nodes(
      result_indices_buffer + result_buffer_size_32 - (graph_degree * search_width),
      result_distances_buffer + result_buffer_size_32 - (graph_degree * search_width),
      *dataset_desc,
      knn_graph,
      graph_degree,
      local_visited_hashmap_ptr,
      visited_hash_bitlen,
      local_traversed_hashmap_ptr,
      traversed_hash_bitlen,
      parent_indices_buffer,
      result_indices_buffer,
      search_width);
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    // Filtering
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                cuvs::neighbors::filtering::none_sample_filter>::value) {
      for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
        if (parent_indices_buffer[p] != invalid_index) {
          const auto parent_id =
            result_indices_buffer[parent_indices_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter(query_id, parent_id)) {
            // If the parent must not be in the resulting top-k list, remove from the parent list
            result_distances_buffer[parent_indices_buffer[p]] = utils::get_max_value<DISTANCE_T>();
            result_indices_buffer[parent_indices_buffer[p]]   = invalid_index;
          }
        }
      }
      __syncthreads();
    }

    iter++;
  }

  // Filtering
  if constexpr (!std::is_same<SAMPLE_FILTER_T,
                              cuvs::neighbors::filtering::none_sample_filter>::value) {
    for (uint32_t i = threadIdx.x; i < result_buffer_size_32; i += blockDim.x) {
      INDEX_T index = result_indices_buffer[i];
      if (index == invalid_index) { continue; }
      index &= ~index_msb_1_mask;
      if (!sample_filter(query_id, index)) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
    __syncthreads();
  }

  // Output search results (1st warp only).
  if (threadIdx.x < 32) {
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
            result_distances_ptr[k] = result_distances_buffer[i];
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

template <class T>
RAFT_KERNEL set_value_batch_kernel(T* const dev_ptr,
                                   const std::size_t ld,
                                   const T val,
                                   const std::size_t count,
                                   const std::size_t batch_size)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count * batch_size) { return; }
  const auto batch_id              = tid / count;
  const auto elem_id               = tid % count;
  dev_ptr[elem_id + ld * batch_id] = val;
}

template <class T>
void set_value_batch(T* const dev_ptr,
                     const std::size_t ld,
                     const T val,
                     const std::size_t count,
                     const std::size_t batch_size,
                     cudaStream_t cuda_stream)
{
  constexpr std::uint32_t block_size = 256;
  const auto grid_size               = (count * batch_size + block_size - 1) / block_size;
  set_value_batch_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(dev_ptr, ld, val, count, batch_size);
}

template <typename DATASET_DESCRIPTOR_T, typename SAMPLE_FILTER_T>
struct search_kernel_config {
  // Search kernel function type. Note that the actual values for the template value
  // parameters do not matter, because they are not part of the function signature. The
  // second to fourth value parameters will be selected by the choose_* functions below.
  using kernel_t = decltype(&search_kernel<128, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>);

  static auto choose_buffer_size(unsigned result_buffer_size, unsigned block_size) -> kernel_t
  {
    if (result_buffer_size <= 64) {
      return search_kernel<64, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>;
    } else if (result_buffer_size <= 128) {
      return search_kernel<128, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>;
    } else if (result_buffer_size <= 256) {
      return search_kernel<256, DATASET_DESCRIPTOR_T, SAMPLE_FILTER_T>;
    }
    THROW("Result buffer size %u larger than max buffer size %u", result_buffer_size, 256);
  }
};

template <typename DataT, typename IndexT, typename DistanceT, typename SampleFilterT>
void select_and_run(const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
                    raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,
                    IndexT* topk_indices_ptr,       // [num_queries, topk]
                    DistanceT* topk_distances_ptr,  // [num_queries, topk]
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
  auto kernel =
    search_kernel_config<dataset_descriptor_base_t<DataT, IndexT, DistanceT>,
                         SampleFilterT>::choose_buffer_size(result_buffer_size, block_size);

  RAFT_CUDA_TRY(
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
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
  RAFT_LOG_DEBUG("Launching kernel with %u threads, (%u, %u) blocks %u smem",
                 block_size,
                 num_cta_per_query,
                 num_queries,
                 smem_size);

  kernel<<<grid_dims, block_dims, smem_size, stream>>>(topk_indices_ptr,
                                                       topk_distances_ptr,
                                                       dataset_desc.dev_ptr(stream),
                                                       queries_ptr,
                                                       graph.data_handle(),
                                                       graph.extent(1),
                                                       ps.num_random_samplings,
                                                       ps.rand_xor_mask,
                                                       dev_seed_ptr,
                                                       num_seeds,
                                                       visited_hash_bitlen,
                                                       traversed_hashmap_ptr,
                                                       traversed_hash_bitlen,
                                                       ps.itopk_size,
                                                       ps.search_width,
                                                       ps.min_iterations,
                                                       ps.max_iterations,
                                                       num_executed_iterations,
                                                       sample_filter);
}

}  // namespace multi_cta_search
}  // namespace cuvs::neighbors::cagra::detail
