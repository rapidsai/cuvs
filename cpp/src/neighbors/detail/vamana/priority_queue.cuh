/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "vamana_structs.cuh"
#include <cstdio>
#include <raft/util/warp_primitives.cuh>

namespace cuvs::neighbors::vamana::detail {

/***************************************************************************************
***************************************************************************************/
/**
 * @defgroup vamana_priority_queue Vamana Priority queue structure
 * @{
 */

/**
 * @brief Priority Queue structure used by Vamana GreedySearch during
 * graph construction.
 *
 * The structure keeps the nearest visited neighbors seen thus far during
 * search and lets us efficiently find the next node to visit during the search.
 * Stores a total of kval pairs, where currently kval must be 2i-1 for some integer
 * i since the heap must be complete.
 * This size is determined during vamana build with the "queue_size" parameter (default 127)
 *
 * The queue and all methods are device-side, with a work group size or 32 (one warp).
 * During search, each warp creates their own queue to search a single query at a time.
 * The device memory pointed to by `vals` is assigned during the call to `initialize`.
 * The Vamana GreedySearch call uses shared memory, but any device-accessible memory is applicable.
 *
 *
 * @tparam IdxT type of the vector indices (represent dataset.extent(0))
 * @tparam AccT type of distances between vectors (accumuator type)
 *
 */
template <typename IdxT, typename AccT>
class priority_queue {
 public:
  int kval;
  int insert_pointer;
  dist_pair<IdxT, AccT>* vals;
  dist_pair<IdxT, AccT> temp;

  int* q_size;
  // Enforce max-heap property on the entries
  __forceinline__ __device__ void heapify()
  {
    int i         = 0;
    int swap_dest = 0;

    while (2 * i + 2 < kval) {
      swap_dest = 2 * i;
      swap_dest +=
        (vals[i].dist > vals[2 * i + 1].dist && vals[2 * i + 2].dist >= vals[2 * i + 1].dist);
      swap_dest +=
        2 * (vals[i].dist > vals[2 * i + 2].dist && vals[2 * i + 1].dist > vals[2 * i + 2].dist);

      if (swap_dest == 2 * i) return;

      swap(&vals[i], &vals[swap_dest]);

      i = swap_dest;
    }
  }

  // Starts the heapify process starting at a particular index
  __forceinline__ __device__ void heapify_at(int idx)
  {
    int i         = idx;
    int swap_dest = 0;

    while (2 * i + 2 < kval) {
      swap_dest = 2 * i;
      swap_dest +=
        (vals[i].dist > vals[2 * i + 1].dist && vals[2 * i + 2].dist <= vals[2 * i + 1].dist);
      swap_dest +=
        2 * (vals[i].dist > vals[2 * i + 2].dist && vals[2 * i + 1].dist < vals[2 * i + 2].dist);

      if (swap_dest == 2 * i) return;

      swap(&vals[i], &vals[swap_dest]);
      i = swap_dest;
    }
  }

  // Heapify from the bottom up, used with insert_back
  __forceinline__ __device__ void heapify_reverse_at(int idx)
  {
    int i         = idx;
    int swap_dest = 0;
    while (i > 0) {
      swap_dest = ((i - 1) / 2);
      if (vals[swap_dest].dist <= vals[i].dist) return;

      swap(&vals[i], &vals[swap_dest]);
      i = swap_dest;
    }
  }

  __device__ void reset()
  {
    *q_size = 0;
    for (int i = 0; i < kval; i++) {
      vals[i].dist = raft::upper_bound<AccT>();
      vals[i].idx  = raft::upper_bound<IdxT>();
    }
  }

  __device__ void initialize(dist_pair<IdxT, AccT>* v, int _kval, int* _q_size)
  {
    vals           = v;
    kval           = _kval;
    insert_pointer = _kval / 2;
    q_size         = _q_size;
    reset();
  }

  // Initialize all nodes of the heap to +infinity
  __device__ void initialize()
  {
    for (int i = 0; i < kval; i++) {
      vals[i].idx  = raft::upper_bound<IdxT>();
      vals[i].dist = raft::upper_bound<AccT>();
    }
  }

  __device__ void write_to_gmem(int* gmem)
  {
    for (int i = 0; i < kval; i++) {
      gmem[i] = vals[i].idx;
    }
  }

  // Replace the root of the heap with new pair
  __device__ void insert(AccT newDist, IdxT newIdx)
  {
    vals[0].dist = newDist;
    vals[0].idx  = newIdx;

    heapify();
  }

  // Replace a specific element in the heap (and maintain heap properties)
  __device__ void insert_at(AccT newDist, IdxT newIdx, int idx)
  {
    vals[idx].dist = newDist;
    vals[idx].idx  = newIdx;

    heapify_at(idx);
  }

  // Return value of the root of the heap (largest value)
  __device__ auto top() -> AccT { return vals[0].dist; }

  __device__ auto top_node() -> IdxT { return vals[0].idx; }

  __device__ void insert_back(AccT newDist, IdxT newIdx)
  {
    if (newDist < vals[insert_pointer].dist) {
      if (vals[insert_pointer].idx == raft::upper_bound<IdxT>()) *q_size += 1;
      vals[insert_pointer].dist = newDist;
      vals[insert_pointer].idx  = newIdx;
      heapify_reverse_at(insert_pointer);
    }
    insert_pointer++;

    if (insert_pointer == kval) insert_pointer = kval / 2;
  }

  // Pop root node off and heapify
  __device__ auto pop() -> dist_pair<IdxT, AccT>
  {
    dist_pair<IdxT, AccT> result;
    result.dist  = vals[0].dist;
    result.idx   = vals[0].idx;
    vals[0].dist = raft::upper_bound<AccT>();
    vals[0].idx  = raft::upper_bound<IdxT>();
    heapify();
    *q_size -= 1;
    return result;
  }
};

/***************************************************************************************
 * node structure used for simplified lists during GreedySearch.
 * Used for other operations like checking for duplicates, etc.
 ****************************************************************************************/
template <typename SUMTYPE>
class __align__(16) node {
 public:
  SUMTYPE distance;
  int nodeid;
};

// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ auto operator<(const node<SUMTYPE>& first, const node<SUMTYPE>& other) -> bool
{
  return first.distance < other.distance;
}

// Less-than operator between two Nodes.
template <typename SUMTYPE>
__host__ __device__ auto operator>(const node<SUMTYPE>& first, const node<SUMTYPE>& other) -> bool
{
  return first.distance > other.distance;
}

template <typename AccT>
__device__ auto check_duplicate(const node<AccT>* pq, const int size, node<AccT> new_node) -> bool
{
  bool found = false;
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    if (pq[i].nodeid == new_node.nodeid) {
      found = true;
      break;
    }
  }

  unsigned mask = raft::ballot(found);

  if (mask == 0) {
    return false;

  } else {
    return true;
  }
}

/*
  Enqueuing a input value into parallel queue with tracker
*/
template <typename SUMTYPE>
__inline__ __device__ void parallel_pq_max_enqueue(node<SUMTYPE>* pq,
                                                   int* size,
                                                   const int pq_size,
                                                   node<SUMTYPE> input_data,
                                                   SUMTYPE* cur_max_val,
                                                   int* max_idx)
{
  if (*size < pq_size) {
    __syncthreads();
    if (threadIdx.x == 0) {
      pq[*size].distance = input_data.distance;
      pq[*size].nodeid   = input_data.nodeid;
      *size              = *size + 1;
      if (input_data.distance > (*cur_max_val)) {
        *cur_max_val = input_data.distance;
        *max_idx     = *size - 1;
      }
    }
    __syncthreads();
    return;
  } else {
    if (input_data.distance >= (*cur_max_val)) {
      __syncthreads();
      return;
    }
    if (threadIdx.x == 0) {
      pq[*max_idx].distance = input_data.distance;
      pq[*max_idx].nodeid   = input_data.nodeid;
    }
    int idx         = 0;
    SUMTYPE max_val = pq[0].distance;

    for (int i = threadIdx.x; i < pq_size; i += 32) {
      if (pq[i].distance > max_val) {
        max_val = pq[i].distance;
        idx     = i;
      }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
      SUMTYPE new_max_val = raft::shfl_up(max_val, offset);
      int new_idx         = raft::shfl_up(idx, offset);
      if (new_max_val > max_val) {
        max_val = new_max_val;
        idx     = new_idx;
      }
    }

    if (threadIdx.x == 31) {
      *max_idx     = idx;
      *cur_max_val = max_val;
    }
  }
  __syncthreads();
}

/*
  Compute the distances between the source vector and all nodes in the neighbor_array and enqueue
  them in the PQ
*/
template <typename T, typename AccT, typename IdxT>
__forceinline__ __device__ void enqueue_all_neighbors(int num_neighbors,
                                                      point<T, AccT>* query_vec,
                                                      const T* vec_ptr,
                                                      int* neighbor_array,
                                                      priority_queue<IdxT, AccT>& heap_queue,
                                                      int dim,
                                                      cuvs::distance::DistanceType metric)
{
  for (int i = 0; i < num_neighbors; i++) {
    AccT dist_out = dist<T, AccT>(
      query_vec->coords, &vec_ptr[(size_t)(neighbor_array[i]) * (size_t)(dim)], dim, metric);

    __syncthreads();
    if (threadIdx.x == 0) { heap_queue.insert_back(dist_out, neighbor_array[i]); }
    __syncthreads();
  }
}

}  // namespace cuvs::neighbors::vamana::detail
