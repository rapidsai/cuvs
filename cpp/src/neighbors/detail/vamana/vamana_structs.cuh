/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <climits>
#include <cstdint>
#include <cstdio>
#include <float.h>
#include <unordered_set>
#include <vector>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::vamana::detail {

/* @defgroup vamana_structures vamana structures
 * @{
 */

#define FULL_BITMASK 0xFFFFFFFF

// Currently supported values for graph_degree.
static const int DEGREE_SIZES[4] = {32, 64, 128, 256};

// Object used to store id,distance combination graph construction operations
template <typename IdxT, typename accT>
struct __align__(16) DistPair {
  accT dist;
  IdxT idx;
};

// Swap the values of two DistPair<SUMTYPE> objects
template <typename IdxT, typename accT>
__device__ __host__ void swap(DistPair<IdxT, accT>* a, DistPair<IdxT, accT>* b)
{
  DistPair<IdxT, accT> temp;
  temp.dist = a->dist;
  temp.idx  = a->idx;
  a->dist   = b->dist;
  a->idx    = b->idx;
  b->dist   = temp.dist;
  b->idx    = temp.idx;
}

// Structure to sort by distance
template <typename IdxT, typename accT>
struct CmpDist {
  __device__ bool operator()(const DistPair<IdxT, accT>& lhs, const DistPair<IdxT, accT>& rhs)
  {
    return lhs.dist < rhs.dist;
  }
};

// Used to sort reverse edges by destination
template <typename IdxT>
struct CmpEdge {
  __device__ bool operator()(const IdxT& lhs, const IdxT& rhs) { return lhs < rhs; }
};

/*********************************************************************
 * Object representing a Dim-dimensional point, with each coordinate
 * represented by a element of datatype T
 * Note, memory is allocated separately and coords set to offsets
 *********************************************************************/
template <typename T, typename SUMTYPE>
class Point {
 public:
  int id;
  int Dim;
  T* coords;

  __host__ __device__ Point& operator=(const Point& other)
  {
    for (int i = 0; i < Dim; i++) {
      coords[i] = other.coords[i];
    }
    id = other.id;
    return *this;
  }
};

/* L2 fallback for low dimension when ILP is not possible */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_SEQ(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  SUMTYPE partial_sum = 0;

  for (int i = threadIdx.x; i < src_vec->Dim; i += blockDim.x) {
    partial_sum = fmaf((src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       (src_vec[0].coords[i] - dst_vec[0].coords[i]),
                       partial_sum);
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(FULL_BITMASK, partial_sum, offset);
  }
  return partial_sum;
}

/* L2 optimized with 2-way ILP for DIM >= 64 */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP2(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  T temp_dst[2]          = {0, 0};
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = threadIdx.x; i < src_vec->Dim; i += 2 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
  }
  partial_sum[0] += partial_sum[1];

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

/* L2 optimized with 4-way ILP for optimal performance for DIM >= 128 */
template <typename T, typename SUMTYPE>
__device__ SUMTYPE l2_ILP4(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  T temp_dst[4]          = {0, 0, 0, 0};
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = threadIdx.x; i < src_vec->Dim; i += 4 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->Dim) temp_dst[1] = dst_vec->coords[i + 32];
    if (i + 64 < src_vec->Dim) temp_dst[2] = dst_vec->coords[i + 64];
    if (i + 96 < src_vec->Dim) temp_dst[3] = dst_vec->coords[i + 96];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->Dim)
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
    if (i + 64 < src_vec->Dim)
      partial_sum[2] = fmaf((src_vec[0].coords[i + 64] - temp_dst[2]),
                            (src_vec[0].coords[i + 64] - temp_dst[2]),
                            partial_sum[2]);
    if (i + 96 < src_vec->Dim)
      partial_sum[3] = fmaf((src_vec[0].coords[i + 96] - temp_dst[3]),
                            (src_vec[0].coords[i + 96] - temp_dst[3]),
                            partial_sum[3]);
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }

  return partial_sum[0];
}

/* Selects ILP optimization level based on dimension */
template <typename T, typename SUMTYPE>
__forceinline__ __device__ SUMTYPE l2(Point<T, SUMTYPE>* src_vec, Point<T, SUMTYPE>* dst_vec)
{
  if (src_vec->Dim >= 128) {
    return l2_ILP4<T, SUMTYPE>(src_vec, dst_vec);
  } else if (src_vec->Dim >= 64) {
    return l2_ILP2<T, SUMTYPE>(src_vec, dst_vec);
  } else {
    return l2_SEQ<T, SUMTYPE>(src_vec, dst_vec);
  }
}

/* Convert vectors to point structure to performance distance comparison */
template <typename T, typename SUMTYPE>
__host__ __device__ SUMTYPE l2(const T* src, const T* dest, int dim)
{
  Point<T, SUMTYPE> src_p;
  src_p.coords = const_cast<T*>(src);
  src_p.Dim    = dim;
  Point<T, SUMTYPE> dest_p;
  dest_p.coords = const_cast<T*>(dest);
  dest_p.Dim    = dim;

  return l2<T, SUMTYPE>(&src_p, &dest_p);
}

// Currently only L2Expanded is supported
template <typename T, typename SUMTYPE>
__host__ __device__ SUMTYPE
dist(const T* src, const T* dest, int dim, cuvs::distance::DistanceType metric)
{
  return l2<T, SUMTYPE>(src, dest, dim);
}

/***************************************************************************************
 * Structure that holds information about and results of a query. Use by both
 * GreedySearch and RobustPrune, as well as reverse edge lists.
 ***************************************************************************************/
template <typename IdxT, typename accT>
struct QueryCandidates {
  IdxT* ids;
  accT* dists;
  int queryId;
  int size;
  int maxSize;

  __device__ void reset()
  {
    for (int i = threadIdx.x; i < maxSize; i += blockDim.x) {
      ids[i]   = raft::upper_bound<IdxT>();
      dists[i] = raft::upper_bound<accT>();
    }
    size = 0;
  }

  // Checks current list to see if a node as previously been visited
  __inline__ __device__ bool check_visited(IdxT target, accT dist)
  {
    __syncthreads();
    __shared__ bool found;
    found = false;
    __syncthreads();

    if (size < maxSize) {
      __syncthreads();
      for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (ids[i] == target) { found = true; }
      }
      __syncthreads();
      if (!found && threadIdx.x == 0) {
        ids[size]   = target;
        dists[size] = dist;
        size++;
      }
      __syncthreads();
    }
    return found;
  }
  // For debugging
  /*
  __inline__ __device__ void print_visited() {
    printf("queryId:%d, size:%d\n", queryId, size);
    for(int i=0; i<size; i++) {
      printf("%d (%f), ", ids[i], dists[i]);
    }
    printf("\n");
  }
  */
};

namespace {

/********************************************************************************************
 * Kernels that work on QueryCandidates objects *
 *******************************************************************************************/
// For debugging
template <typename accT, typename IdxT = uint32_t>
__global__ void print_query_results(void* query_list_ptr, int count)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = 0; i < count; i++) {
    query_list[i].print_visited();
  }
}

// Initialize a list of QueryCandidates objects: assign memory to mpointers and initialize values
template <typename IdxT, typename accT>
__global__ void init_query_candidate_list(QueryCandidates<IdxT, accT>* query_list,
                                          IdxT* visited_id_ptr,
                                          accT* visited_dist_ptr,
                                          int num_queries,
                                          int maxSize,
                                          int extra_queries_in_list = 0)
{
  IdxT* ids_ptr  = static_cast<IdxT*>(visited_id_ptr);
  accT* dist_ptr = static_cast<accT*>(visited_dist_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries * maxSize;
       i += blockDim.x * gridDim.x) {
    ids_ptr[i]  = raft::upper_bound<IdxT>();
    dist_ptr[i] = raft::upper_bound<accT>();
  }

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries + extra_queries_in_list;
       i += blockDim.x * gridDim.x) {
    query_list[i].maxSize = maxSize;
    query_list[i].size    = 0;
    query_list[i].ids     = &ids_ptr[i * (size_t)(maxSize)];
    query_list[i].dists   = &dist_ptr[i * (size_t)(maxSize)];
  }
}

// Copy query ID values from input array
template <typename IdxT, typename accT>
__global__ void set_query_ids(void* query_list_ptr, IdxT* d_query_ids, int step_size)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < step_size; i += blockDim.x * gridDim.x) {
    query_list[i].queryId = d_query_ids[i];
    query_list[i].size    = 0;
  }
}

// Compute prefix sums on sizes. Currently only works with 1 thread
// TODO replace with parallel version
template <typename accT, typename IdxT = uint32_t>
__global__ void prefix_sums_sizes(QueryCandidates<IdxT, accT>* query_list,
                                  int num_queries,
                                  int* total_edges)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < num_queries + 1; i++) {
      sum += query_list[i].size;
      query_list[i].size = sum - query_list[i].size;  // exclusive prefix sum
    }
    *total_edges = query_list[num_queries].size;
  }
}

// Device fcn to have a threadblock copy coordinates into shared memory
template <typename T, typename accT>
__device__ void update_shared_point(
  Point<T, accT>* shared_point, const T* data_ptr, int id, int dim, int idx)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Device fcn to have a threadblock copy coordinates into shared memory
template <typename T, typename accT>
__device__ void update_shared_point(Point<T, accT>* shared_point,
                                    const T* data_ptr,
                                    int id,
                                    int dim)
{
  shared_point->id  = id;
  shared_point->Dim = dim;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Update the graph from the results of the query list (or reverse edge list)
template <typename accT, typename IdxT = uint32_t>
__global__ void write_graph_edges_kernel(raft::device_matrix_view<IdxT, int64_t> graph,
                                         void* query_list_ptr,
                                         int degree,
                                         int num_queries)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      graph(query_list[i].queryId, j) = query_list[i].ids[j];
    }
  }
}

// Create src and dest edge lists used to sort and create reverse edges
template <typename accT, typename IdxT = uint32_t>
__global__ void create_reverse_edge_list(void* query_list_ptr,
                                         int num_queries,
                                         int degree,
                                         IdxT* edge_src,
                                         DistPair<IdxT, accT>* edge_dest)
{
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries;
       i += blockDim.x * gridDim.x) {
    int read_idx   = i * query_list[i].maxSize;
    int cand_count = query_list[i + 1].size - query_list[i].size;

    for (int j = 0; j < cand_count; j++) {
      edge_src[query_list[i].size + j]       = query_list[i].queryId;
      edge_dest[query_list[i].size + j].idx  = query_list[i].ids[j];
      edge_dest[query_list[i].size + j].dist = query_list[i].dists[j];
    }
  }
}

// Populate reverse edge QueryCandidates structure based on sorted edge list and unique indices
// values
template <typename T, typename accT, typename IdxT = uint32_t>
__global__ void populate_reverse_list_struct(QueryCandidates<IdxT, accT>* reverse_list,
                                             IdxT* edge_src,
                                             IdxT* edge_dest,
                                             int* unique_indices,
                                             int unique_dests,
                                             int total_edges,
                                             int N,
                                             int rev_start,
                                             int reverse_batch)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < reverse_batch;
       i += blockDim.x * gridDim.x) {
    reverse_list[i].queryId = edge_dest[unique_indices[i + rev_start]];
    if (rev_start + i == unique_dests - 1) {
      reverse_list[i].size = total_edges - unique_indices[i + rev_start];
    } else {
      reverse_list[i].size = unique_indices[i + rev_start + 1] - unique_indices[i + rev_start];
    }
    if (reverse_list[i].size > reverse_list[i].maxSize) {
      reverse_list[i].size = reverse_list[i].maxSize;
    }

    for (int j = 0; j < reverse_list[i].size; j++) {
      reverse_list[i].ids[j] = edge_src[unique_indices[i + rev_start] + j];
    }
    for (int j = reverse_list[i].size; j < reverse_list[i].maxSize; j++) {
      reverse_list[i].ids[j]   = raft::upper_bound<IdxT>();
      reverse_list[i].dists[j] = raft::upper_bound<accT>();
    }
  }
}

// Recompute distances of reverse list. Allows us to avoid keeping distances during sort
template <typename T,
          typename accT,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
__global__ void recompute_reverse_dists(
  QueryCandidates<IdxT, accT>* reverse_list,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int unique_dests,
  cuvs::distance::DistanceType metric)
{
  int dim          = dataset.extent(1);
  const T* vec_ptr = dataset.data_handle();

  for (int i = blockIdx.x; i < unique_dests; i += gridDim.x) {
    for (int j = 0; j < reverse_list[i].size; j++) {
      reverse_list[i].dists[j] =
        dist<T, accT>(&vec_ptr[(size_t)(reverse_list[i].queryId) * (size_t)dim],
                      &vec_ptr[(size_t)(reverse_list[i].ids[j]) * (size_t)dim],
                      dim,
                      metric);
    }
  }
}

}  // namespace

/**
 * @}
 */

}  // namespace cuvs::neighbors::vamana::detail
