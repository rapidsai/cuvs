/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdio>
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
static const int kDegreeSizes[4] = {32, 64, 128, 256};

// Object used to store id,distance combination graph construction operations
template <typename IdxT, typename Acct>
struct __align__(16) dist_pair {
  Acct dist;
  IdxT idx;
};

// Swap the values of two dist_pair<SUMTYPE> objects
template <typename IdxT, typename Acct>
__device__ __host__ void swap(dist_pair<IdxT, Acct>* a, dist_pair<IdxT, Acct>* b)
{
  dist_pair<IdxT, Acct> temp;
  temp.dist = a->dist;
  temp.idx  = a->idx;
  a->dist   = b->dist;
  a->idx    = b->idx;
  b->dist   = temp.dist;
  b->idx    = temp.idx;
}

// Structure to sort by distance
template <typename IdxT, typename Acct>
struct cmp_dist {
  __device__ auto operator()(const dist_pair<IdxT, Acct>& lhs, const dist_pair<IdxT, Acct>& rhs)
    -> bool
  {
    return lhs.dist < rhs.dist;
  }
};

// Used to sort reverse edges by destination
template <typename IdxT>
struct cmp_edge {
  __device__ auto operator()(const IdxT& lhs, const IdxT& rhs) -> bool { return lhs < rhs; }
};

/*********************************************************************
 * Object representing a dim-dimensional point, with each coordinate
 * represented by a element of datatype T
 * Note, memory is allocated separately and coords set to offsets
 *********************************************************************/
template <typename T, typename SUMTYPE>
class point {
 public:
  int id;
  int dim;
  T* coords;

  __host__ __device__ auto operator=(const point& other) -> point&
  {
    for (int i = 0; i < dim; i++) {
      coords[i] = other.coords[i];
    }
    id = other.id;
    return *this;
  }
};

/* L2 fallback for low dimension when ILP is not possible */
template <typename T, typename SUMTYPE>
__device__ auto l2_seq(point<T, SUMTYPE>* src_vec, point<T, SUMTYPE>* dst_vec) -> SUMTYPE
{
  SUMTYPE partial_sum = 0;

  for (int i = threadIdx.x; i < src_vec->dim; i += blockDim.x) {
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
__device__ auto l2_ilp2(point<T, SUMTYPE>* src_vec, point<T, SUMTYPE>* dst_vec) -> SUMTYPE
{
  T temp_dst[2]          = {0, 0};
  SUMTYPE partial_sum[2] = {0, 0};
  for (int i = threadIdx.x; i < src_vec->dim; i += 2 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->dim) temp_dst[1] = dst_vec->coords[i + 32];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->dim) {
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
    }
  }
  partial_sum[0] += partial_sum[1];

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }
  return partial_sum[0];
}

/* L2 optimized with 4-way ILP for optimal performance for DIM >= 128 */
template <typename T, typename SUMTYPE>
__device__ auto l2_ilp4(point<T, SUMTYPE>* src_vec, point<T, SUMTYPE>* dst_vec) -> SUMTYPE
{
  T temp_dst[4]          = {0, 0, 0, 0};
  SUMTYPE partial_sum[4] = {0, 0, 0, 0};
  for (int i = threadIdx.x; i < src_vec->dim; i += 4 * blockDim.x) {
    temp_dst[0] = dst_vec->coords[i];
    if (i + 32 < src_vec->dim) temp_dst[1] = dst_vec->coords[i + 32];
    if (i + 64 < src_vec->dim) temp_dst[2] = dst_vec->coords[i + 64];
    if (i + 96 < src_vec->dim) temp_dst[3] = dst_vec->coords[i + 96];

    partial_sum[0] = fmaf(
      (src_vec[0].coords[i] - temp_dst[0]), (src_vec[0].coords[i] - temp_dst[0]), partial_sum[0]);
    if (i + 32 < src_vec->dim) {
      partial_sum[1] = fmaf((src_vec[0].coords[i + 32] - temp_dst[1]),
                            (src_vec[0].coords[i + 32] - temp_dst[1]),
                            partial_sum[1]);
    }
    if (i + 64 < src_vec->dim) {
      partial_sum[2] = fmaf((src_vec[0].coords[i + 64] - temp_dst[2]),
                            (src_vec[0].coords[i + 64] - temp_dst[2]),
                            partial_sum[2]);
    }
    if (i + 96 < src_vec->dim) {
      partial_sum[3] = fmaf((src_vec[0].coords[i + 96] - temp_dst[3]),
                            (src_vec[0].coords[i + 96] - temp_dst[3]),
                            partial_sum[3]);
    }
  }
  partial_sum[0] += partial_sum[1] + partial_sum[2] + partial_sum[3];

  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum[0] += __shfl_down_sync(FULL_BITMASK, partial_sum[0], offset);
  }

  return partial_sum[0];
}

/* Selects ILP optimization level based on dimension */
template <typename T, typename SUMTYPE>
__forceinline__ __device__ auto l2(point<T, SUMTYPE>* src_vec, point<T, SUMTYPE>* dst_vec)
  -> SUMTYPE
{
  if (src_vec->dim >= 128) {
    return l2_ilp4<T, SUMTYPE>(src_vec, dst_vec);
  } else if (src_vec->dim >= 64) {
    return l2_ilp2<T, SUMTYPE>(src_vec, dst_vec);
  } else {
    return l2_seq<T, SUMTYPE>(src_vec, dst_vec);
  }
}

/* Convert vectors to point structure to performance distance comparison */
template <typename T, typename SUMTYPE>
__host__ __device__ auto l2(const T* src, const T* dest, int dim) -> SUMTYPE
{
  point<T, SUMTYPE> src_p;
  src_p.coords = const_cast<T*>(src);
  src_p.dim    = dim;
  point<T, SUMTYPE> dest_p;
  dest_p.coords = const_cast<T*>(dest);
  dest_p.dim    = dim;

  return l2<T, SUMTYPE>(&src_p, &dest_p);
}

// Currently only L2Expanded is supported
template <typename T, typename SUMTYPE>
__host__ __device__ auto dist(const T* src,
                              const T* dest,
                              int dim,
                              cuvs::distance::DistanceType metric) -> SUMTYPE
{
  return l2<T, SUMTYPE>(src, dest, dim);
}

/***************************************************************************************
 * Structure that holds information about and results of a query. Use by both
 * GreedySearch and RobustPrune, as well as reverse edge lists.
 ***************************************************************************************/
template <typename IdxT, typename Acct>
struct query_candidates {
  IdxT* ids;
  Acct* dists;
  int query_id;
  int size;
  int max_size;

  __device__ void reset()
  {
    for (int i = threadIdx.x; i < max_size; i += blockDim.x) {
      ids[i]   = raft::upper_bound<IdxT>();
      dists[i] = raft::upper_bound<Acct>();
    }
    size = 0;
  }

  // Checks current list to see if a node as previously been visited
  __inline__ __device__ auto check_visited(IdxT target, Acct dist) -> bool
  {
    __syncthreads();
    __shared__ bool found;
    found = false;
    __syncthreads();

    if (size < max_size) {
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
    printf("query_id:%d, size:%d\n", query_id, size);
    for(int i=0; i<size; i++) {
      printf("%d (%f), ", ids[i], dists[i]);
    }
    printf("\n");
  }
  */
};

namespace {

/********************************************************************************************
 * Kernels that work on query_candidates objects *
 *******************************************************************************************/
// For debugging
template <typename Acct, typename IdxT = uint32_t>
__global__ void print_query_results(void* query_list_ptr, int count)
{
  auto* query_list = static_cast<query_candidates<IdxT, Acct>*>(query_list_ptr);

  for (int i = 0; i < count; i++) {
    query_list[i].print_visited();
  }
}

// Initialize a list of query_candidates objects: assign memory to mpointers and initialize values
template <typename IdxT, typename Acct>
__global__ void init_query_candidate_list(query_candidates<IdxT, Acct>* query_list,
                                          IdxT* visited_id_ptr,
                                          Acct* visited_dist_ptr,
                                          int num_queries,
                                          int max_size,
                                          int extra_queries_in_list = 0)
{
  IdxT* ids_ptr  = static_cast<IdxT*>(visited_id_ptr);
  Acct* dist_ptr = static_cast<Acct*>(visited_dist_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries * max_size;
       i += blockDim.x * gridDim.x) {
    ids_ptr[i]  = raft::upper_bound<IdxT>();
    dist_ptr[i] = raft::upper_bound<Acct>();
  }

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries + extra_queries_in_list;
       i += blockDim.x * gridDim.x) {
    query_list[i].max_size = max_size;
    query_list[i].size     = 0;
    query_list[i].ids      = &ids_ptr[i * (size_t)(max_size)];
    query_list[i].dists    = &dist_ptr[i * (size_t)(max_size)];
  }
}

// Copy query ID values from input array
template <typename IdxT, typename Acct>
__global__ void set_query_ids(void* query_list_ptr, IdxT* d_query_ids, int step_size)
{
  auto* query_list = static_cast<query_candidates<IdxT, Acct>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < step_size; i += blockDim.x * gridDim.x) {
    query_list[i].query_id = d_query_ids[i];
    query_list[i].size     = 0;
  }
}

// Compute prefix sums on sizes. Currently only works with 1 thread
// TODO(snanditale): replace with parallel version
template <typename Acct, typename IdxT = uint32_t>
__global__ void prefix_sums_sizes(query_candidates<IdxT, Acct>* query_list,
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
template <typename T, typename Acct>
__device__ void update_shared_point(
  point<T, Acct>* shared_point, const T* data_ptr, int id, int dim, int idx)
{
  shared_point->id  = id;
  shared_point->dim = dim;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Device fcn to have a threadblock copy coordinates into shared memory
template <typename T, typename Acct>
__device__ void update_shared_point(point<T, Acct>* shared_point,
                                    const T* data_ptr,
                                    int id,
                                    int dim)
{
  shared_point->id  = id;
  shared_point->dim = dim;
  for (size_t i = threadIdx.x; i < dim; i += blockDim.x) {
    shared_point->coords[i] = data_ptr[(size_t)(id) * (size_t)(dim) + i];
  }
}

// Update the graph from the results of the query list (or reverse edge list)
template <typename Acct, typename IdxT = uint32_t>
__global__ void write_graph_edges_kernel(raft::device_matrix_view<IdxT, int64_t> graph,
                                         void* query_list_ptr,
                                         int degree,
                                         int num_queries)
{
  query_candidates<IdxT, Acct>* query_list =
    static_cast<query_candidates<IdxT, Acct>*>(query_list_ptr);

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    for (int j = threadIdx.x; j < query_list[i].size; j += blockDim.x) {
      graph(query_list[i].query_id, j) = query_list[i].ids[j];
    }
  }
}

// Create src and dest edge lists used to sort and create reverse edges
template <typename Acct, typename IdxT = uint32_t>
__global__ void create_reverse_edge_list(void* query_list_ptr,
                                         int num_queries,
                                         int degree,
                                         IdxT* edge_src,
                                         dist_pair<IdxT, Acct>* edge_dest)
{
  auto* query_list = static_cast<query_candidates<IdxT, Acct>*>(query_list_ptr);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_queries;
       i += blockDim.x * gridDim.x) {
    int read_idx   = i * query_list[i].max_size;
    int cand_count = query_list[i + 1].size - query_list[i].size;

    for (int j = 0; j < cand_count; j++) {
      edge_src[query_list[i].size + j]       = query_list[i].query_id;
      edge_dest[query_list[i].size + j].idx  = query_list[i].ids[j];
      edge_dest[query_list[i].size + j].dist = query_list[i].dists[j];
    }
  }
}

// Populate reverse edge query_candidates structure based on sorted edge list and unique indices
// values
template <typename T, typename Acct, typename IdxT = uint32_t>
__global__ void populate_reverse_list_struct(query_candidates<IdxT, Acct>* reverse_list,
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
    reverse_list[i].query_id = edge_dest[unique_indices[i + rev_start]];
    if (rev_start + i == unique_dests - 1) {
      reverse_list[i].size = total_edges - unique_indices[i + rev_start];
    } else {
      reverse_list[i].size = unique_indices[i + rev_start + 1] - unique_indices[i + rev_start];
    }
    if (reverse_list[i].size > reverse_list[i].max_size) {
      reverse_list[i].size = reverse_list[i].max_size;
    }

    for (int j = 0; j < reverse_list[i].size; j++) {
      reverse_list[i].ids[j] = edge_src[unique_indices[i + rev_start] + j];
    }
    for (int j = reverse_list[i].size; j < reverse_list[i].max_size; j++) {
      reverse_list[i].ids[j]   = raft::upper_bound<IdxT>();
      reverse_list[i].dists[j] = raft::upper_bound<Acct>();
    }
  }
}

// Recompute distances of reverse list. Allows us to avoid keeping distances during sort
template <typename T,
          typename Acct,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
__global__ void recompute_reverse_dists(
  query_candidates<IdxT, Acct>* reverse_list,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  int unique_dests,
  cuvs::distance::DistanceType metric)
{
  int dim          = dataset.extent(1);
  const T* vec_ptr = dataset.data_handle();

  for (int i = blockIdx.x; i < unique_dests; i += gridDim.x) {
    for (int j = 0; j < reverse_list[i].size; j++) {
      reverse_list[i].dists[j] =
        dist<T, Acct>(&vec_ptr[(size_t)(reverse_list[i].query_id) * (size_t)dim],
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
