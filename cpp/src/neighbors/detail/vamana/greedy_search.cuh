/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "macros.cuh"
#include "priority_queue.cuh"
#include "vamana_structs.cuh"
#include <cub/cub.cuh>
#include <cuvs/neighbors/vamana.hpp>

#include <cuvs/distance/distance.hpp>
#include <raft/util/warp_primitives.cuh>
#include <rmm/resource_ref.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

namespace cuvs::neighbors::experimental::vamana::detail {

/* @defgroup greedy_search_detail greedy search
 * @{
 */

/* Combines edge and candidate lists, removes duplicates, and sorts by distance
 * Uses CUB primitives, so needs to be templated. Called with Macros for supported sizes above */
template <typename accT, typename IdxT, int CANDS>
__forceinline__ __device__ void sort_visited(
  QueryCandidates<IdxT, accT>* query,
  typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (CANDS / 32)>::TempStorage* sort_mem)
{
  const int ELTS   = CANDS / 32;
  using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, ELTS>;
  DistPair<IdxT, accT> tmp[ELTS];
  for (int i = 0; i < ELTS; i++) {
    tmp[i].idx  = query->ids[ELTS * threadIdx.x + i];
    tmp[i].dist = query->dists[ELTS * threadIdx.x + i];
  }

  __syncthreads();
  BlockSortT(*sort_mem).Sort(tmp, CmpDist());
  __syncthreads();

  for (int i = 0; i < ELTS; i++) {
    query->ids[ELTS * threadIdx.x + i]   = tmp[i].idx;
    query->dists[ELTS * threadIdx.x + i] = tmp[i].dist;
  }
  __syncthreads();
}

namespace {

/********************************************************************************************
  GPU kernel to perform a batched GreedySearch on a graph. Since this is used for
  Vamana construction, the entire visited list is kept and stored within the query_list.
  Input - graph with edge lists, dataset vectors, query_list_ptr with the ids of dataset
          vectors to be searched. All inputs, including dataset,  must be device accessible.

  Output - the id and dist lists in query_list_ptr will be updated with the nodes visited
           during the GreedySearch.
**********************************************************************************************/
template <typename T,
          typename accT,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
__global__ void GreedySearchKernel(
  raft::device_matrix_view<IdxT, int64_t> graph,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  void* query_list_ptr,
  int num_queries,
  int medoid_id,
  int topk,
  cuvs::distance::DistanceType metric,
  int max_queue_size,
  int sort_smem_size)
{
  int n      = dataset.extent(0);
  int dim    = dataset.extent(1);
  int degree = graph.extent(1);

  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  static __shared__ int topk_q_size;
  static __shared__ int cand_q_size;
  static __shared__ accT cur_k_max;
  static __shared__ int k_max_idx;

  static __shared__ Point<T, accT> s_query;


  union ShmemLayout {
    // All blocksort sizes have same alignment (16)
    typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, 1>::TempStorage sort_mem;
    T coords;
    Node<accT> topk_pq;
    int neighborhood_arr;
    DistPair<IdxT, accT> candidate_queue;
  };

  int align_padding = (((dim-1)/alignof(ShmemLayout))+1)*alignof(ShmemLayout) - dim;

  // Dynamic shared memory used for blocksort, temp vector storage, and neighborhood list
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];

  size_t smem_offset = sort_smem_size;  // temp sorting memory takes first chunk

  T* s_coords = reinterpret_cast<T*>(&smem[smem_offset]);
  smem_offset += (dim+align_padding) * sizeof(T);

  Node<accT>* topk_pq = reinterpret_cast<Node<accT>*>(&smem[smem_offset]);
  smem_offset += topk * sizeof(Node<accT>);

  int* neighbor_array = reinterpret_cast<int*>(&smem[smem_offset]);
  smem_offset += degree * sizeof(int);

  DistPair<IdxT, accT>* candidate_queue_smem =
    reinterpret_cast<DistPair<IdxT, accT>*>(&smem[smem_offset]);

  s_query.coords = s_coords;
  s_query.Dim    = dim;

  PriorityQueue<IdxT, accT> heap_queue;

  if (threadIdx.x == 0) {
    heap_queue.initialize(candidate_queue_smem, max_queue_size, &cand_q_size);
  }

  static __shared__ int num_neighbors;

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    __syncthreads();

    // resetting visited list
    query_list[i].reset();

    // storing the current query vector into shared memory
    update_shared_point<T, accT>(&s_query, &dataset(0, 0), query_list[i].queryId, dim);

    if (threadIdx.x == 0) {
      topk_q_size = 0;
      cand_q_size = 0;
      s_query.id  = query_list[i].queryId;
      cur_k_max   = 0;
      k_max_idx   = 0;
      heap_queue.reset();
    }

    __syncthreads();

    Point<T, accT>* query_vec;

    // Just start from medoid every time, rather than multiple set_ups
    query_vec        = &s_query;
    query_vec->Dim   = dim;
    const T* medoid  = &dataset((size_t)medoid_id, 0);
    accT medoid_dist = dist<T, accT>(query_vec->coords, medoid, dim, metric);

    if (threadIdx.x == 0) { heap_queue.insert_back(medoid_dist, medoid_id); }
    __syncthreads();
    
    while (cand_q_size != 0) {
      __syncthreads();

      int cand_num;
      accT cur_distance;
      if (threadIdx.x == 0) {
        Node<accT> test_cand;
        DistPair<IdxT, accT> test_cand_out = heap_queue.pop();
        test_cand.distance                 = test_cand_out.dist;
        test_cand.nodeid                   = test_cand_out.idx;
        cand_num                           = test_cand.nodeid;
        cur_distance                       = test_cand_out.dist;
      }
      __syncthreads();

      cand_num = raft::shfl(cand_num, 0);

      __syncthreads();

      if (query_list[i].check_visited(cand_num, cur_distance)) { continue; }

      cur_distance = raft::shfl(cur_distance, 0);

      // stop condition for the graph traversal process
      bool done      = false;
      bool pass_flag = false;

      if (topk_q_size == topk) {
        // Check the current node with the worst candidate in top-k queue
        if (threadIdx.x == 0) {
          if (cur_k_max <= cur_distance) { done = true; }
        }

        done = raft::shfl(done, 0);
        if (done) {
          if (query_list[i].size < topk) {
            pass_flag = true;
          }

          else if (query_list[i].size >= topk) {
            break;
          }
        }
      }

      // The current node is closer to the query vector than the worst candidate in top-K queue, so
      // enquee the current node in top-k queue
      Node<accT> new_cand;
      new_cand.distance = cur_distance;
      new_cand.nodeid   = cand_num;

      if (check_duplicate(topk_pq, topk_q_size, new_cand) == false) {
        if (!pass_flag) {
          parallel_pq_max_enqueue<accT>(
            topk_pq, &topk_q_size, topk, new_cand, &cur_k_max, &k_max_idx);

          __syncthreads();
        }
      } else {
        // already visited
        continue;
      }

      num_neighbors = degree;
      __syncthreads();

      for (size_t j = threadIdx.x; j < degree; j += blockDim.x) {
        // Load neighbors from the graph array and store them in neighbor array (shared memory)
        neighbor_array[j] = graph(cand_num, j);
        if (neighbor_array[j] == raft::upper_bound<IdxT>())
          atomicMin(&num_neighbors, (int)j);  // warp-wide min to find the number of neighbors
      }

      // computing distances between the query vector and neighbor vectors then enqueue in priority
      // queue.
      enqueue_all_neighbors<T, accT, IdxT>(
        num_neighbors, query_vec, &dataset(0, 0), neighbor_array, heap_queue, dim, metric);

      __syncthreads();

    }  // End cand_q_size != 0 loop

    bool self_found = false;
    // Remove self edges
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      if (query_list[i].ids[j] == query_vec->id) {
        query_list[i].dists[j] = raft::upper_bound<accT>();
        query_list[i].ids[j]   = raft::upper_bound<IdxT>();
        self_found             = true;  // Flag to reduce size by 1
      }
    }

    for (int j = query_list[i].size + threadIdx.x; j < query_list[i].maxSize; j += blockDim.x) {
      query_list[i].ids[j]   = raft::upper_bound<IdxT>();
      query_list[i].dists[j] = raft::upper_bound<accT>();
    }

    __syncthreads();
    if (self_found) query_list[i].size--;

    SEARCH_SELECT_SORT(topk);
  }

  return;
}

}  // namespace

/**
 * @}
 */

}  // namespace cuvs::neighbors::experimental::vamana::detail
