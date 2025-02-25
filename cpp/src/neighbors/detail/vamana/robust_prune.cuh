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

#include <cub/cub.cuh>
#include <thrust/sort.h>

#include <raft/util/cuda_dev_essentials.cuh>

#include "macros.cuh"
#include "vamana_structs.cuh"

namespace cuvs::neighbors::vamana::detail {

// Load candidates (from query) and previous edges (from nbh_list) into registers (tmp) spanning
// warp
template <typename accT, typename IdxT = uint32_t, int DEG, int CANDS>
__forceinline__ __device__ void load_to_registers(DistPair<IdxT, accT>* tmp,
                                                  QueryCandidates<IdxT, accT>* query,
                                                  DistPair<IdxT, accT>* nbh_list)
{
  int cands_per_thread = CANDS / 32;
  for (int i = 0; i < cands_per_thread; i++) {
    tmp[i].idx  = query->ids[cands_per_thread * threadIdx.x + i];
    tmp[i].dist = query->dists[cands_per_thread * threadIdx.x + i];

    if (cands_per_thread * threadIdx.x + i >= query->size) {
      tmp[i].idx  = raft::upper_bound<IdxT>();
      tmp[i].dist = raft::upper_bound<accT>();
    }
  }
  int nbh_per_thread = DEG / 32;
  for (int i = 0; i < nbh_per_thread; i++) {
    tmp[cands_per_thread + i] = nbh_list[nbh_per_thread * threadIdx.x + i];
  }
}

/* Combines edge and candidate lists, removes duplicates, and sorts by distance
 * Uses CUB primitives, so needs to be templated. Called with Macros for supported sizes above */
template <typename accT, typename IdxT, int DEG, int CANDS>
__forceinline__ __device__ void sort_edges_and_cands(
  DistPair<IdxT, accT>* new_nbh_list,
  QueryCandidates<IdxT, accT>* query,
  typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, (DEG + CANDS) / 32>::TempStorage* sort_mem)
{
  const int ELTS   = (DEG + CANDS) / 32;
  using BlockSortT = cub::BlockMergeSort<DistPair<IdxT, accT>, 32, ELTS>;
  DistPair<IdxT, accT> tmp[ELTS];

  load_to_registers<accT, IdxT, DEG, CANDS>(tmp, query, new_nbh_list);

  __syncthreads();
  BlockSortT(*sort_mem).Sort(tmp, CmpDist());
  __syncthreads();

  // Mark duplicates and re-sort
  // Copy last element over and shuffle to check for duplicate between threads
  new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)] = tmp[ELTS - 1];
  if (tmp[ELTS - 1].idx == tmp[ELTS - 2].idx) {
    new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].idx  = raft::upper_bound<IdxT>();
    new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].dist = raft::upper_bound<accT>();
  }
  __shfl_up_sync(0xffffffff, tmp[ELTS - 1].idx, 1);
  __syncthreads();

  for (int i = ELTS - 2; i > 0; i--) {
    if (tmp[i].idx == tmp[i - 1].idx) {
      tmp[i].idx  = raft::upper_bound<IdxT>();
      tmp[i].dist = raft::upper_bound<accT>();
    }
  }
  if (threadIdx.x == 0) {
    if (tmp[0].idx == tmp[ELTS - 1].idx) {
      tmp[0].idx  = raft::upper_bound<IdxT>();
      tmp[0].dist = raft::upper_bound<accT>();
    }
  }

  tmp[ELTS - 1].idx =
    new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].idx;  // copy back to tmp for re-shuffling
  tmp[ELTS - 1].dist = new_nbh_list[ELTS * threadIdx.x + (ELTS - 1)].dist;

  __syncthreads();
  BlockSortT(*sort_mem).Sort(tmp, CmpDist());
  __syncthreads();

  for (int i = 0; i < ELTS; i++) {
    new_nbh_list[ELTS * threadIdx.x + i].idx  = tmp[i].idx;
    new_nbh_list[ELTS * threadIdx.x + i].dist = tmp[i].dist;
  }
  __syncthreads();
}

namespace {

/********************************************************************************************
  GPU kernel for RobustPrune operation for Vamana graph creation
  Input - *graph to be an edgelist of degree number of edges per vector,
  query_list should contain the list of visited nodes during GreedySearch.
  All inputs, including dataset, must be device accessible.

  Output - candidate_ptr contains the new set of *degree* new neighbors that each node
           should have.
**********************************************************************************************/
template <typename T,
          typename accT,
          typename IdxT     = uint32_t,
          typename Accessor = raft::host_device_accessor<std::experimental::default_accessor<T>,
                                                         raft::memory_type::host>>
__global__ void RobustPruneKernel(
  raft::device_matrix_view<IdxT, int64_t> graph,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  void* query_list_ptr,
  int num_queries,
  int visited_size,
  cuvs::distance::DistanceType metric,
  float alpha,
  int sort_smem_size)
{
  int n      = dataset.extent(0);
  int dim    = dataset.extent(1);
  int degree = graph.extent(1);
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  union ShmemLayout {
    // All blocksort sizes have same alignment (16)
//    typename cub::BlockMergeSort<DistPair<IdxT, accT>, 32, 3>::TempStorage sort_mem;
    T coords;
    DistPair<IdxT, accT> nbh_list;
  };

  // Dynamic shared memory used for blocksort, temp vector storage, and neighborhood list
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];

  int align_padding = raft::alignTo<int>(dim, alignof(ShmemLayout)) - dim;

  T* s_coords                        = reinterpret_cast<T*>(&smem[sort_smem_size]);
  DistPair<IdxT, accT>* new_nbh_list = reinterpret_cast<DistPair<IdxT, accT>*>(
    &smem[(dim + align_padding) * sizeof(T) + sort_smem_size]);

  static __shared__ Point<T, accT> s_query;
  s_query.coords = s_coords;
  s_query.Dim    = dim;
  static __shared__ int prev_edges;
  static __shared__ accT graphDist;

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    int queryId = query_list[i].queryId;

    update_shared_point<T, accT>(&s_query, &dataset(0, 0), queryId, dim);

    int graphIdx=0;
    int listIdx=0;
    int res_size=degree;

    // Count total valid edge candidates
    __syncthreads();
    if(threadIdx.x==0) {
      prev_edges=degree;
      for(int j=0; j<degree; j++) {
        if(graph(queryId, j) == raft::upper_bound<IdxT>()) {
          prev_edges = j;
          break;
        }
      }
    }
    __syncthreads();

    DistPair<IdxT,accT> next_cand;
    for(int outIdx = 0; outIdx < degree;) {

      if(graphIdx < degree && graph(queryId, graphIdx) == raft::upper_bound<IdxT>()) {
        graphIdx = degree;
      }
      if(listIdx < visited_size && query_list[i].ids[listIdx] == raft::upper_bound<IdxT>()) {
        listIdx = visited_size;
      }

    // Get next candidate vector for list
      if(graphIdx >= degree) {
        if(listIdx >= visited_size) { // Fill remaining list if no candidates
          if(res_size > outIdx) res_size = outIdx; // Set result size
          new_nbh_list[outIdx].idx = raft::upper_bound<IdxT>();
          new_nbh_list[outIdx].dist = raft::upper_bound<accT>();
          outIdx++;
          __syncthreads();
          continue;
        }
        else {
          next_cand.idx = query_list[i].ids[listIdx];
          next_cand.dist = query_list[i].dists[listIdx];
          listIdx++;
        }
      }
      else if(listIdx >= visited_size) {
        next_cand.idx = graph(queryId, graphIdx);
        accT tempDist = dist<T, accT>(s_query.coords, &dataset((size_t)graph(queryId, graphIdx), 0), dim, metric);
        if(threadIdx.x==0) graphDist = tempDist;
        __syncthreads();
        graphIdx++;
      }
      else {
        accT listDist = query_list[i].dists[listIdx];

        accT tempDist = dist<T, accT>(s_query.coords, &dataset((size_t)graph(queryId, graphIdx), 0), dim, metric);
        if(threadIdx.x==0) graphDist = tempDist;
        __syncthreads();

        if(listDist <= graphDist) {
          next_cand.idx = query_list[i].ids[listIdx];
          next_cand.dist = listDist;

          if(graph(queryId, graphIdx) == query_list[i].ids[listIdx]) { // Duplicate found!
            graphIdx++; // Skip the duplicate
          }
          listIdx++;
        }
        else {
          next_cand.idx = graph(queryId, graphIdx);
          next_cand.dist = graphDist;
          graphIdx++;
        }
      }

      // Verify candidate doesn't violate current outlist
      bool good=true;
      // Only prune if have too many remaining edge candidates
//      if((prev_edges + query_list[i].size) - (graphIdx + listIdx) > (degree - outIdx)) { 
      if((prev_edges + query_list[i].size) > degree ) {
//        for(int j=0; j<outIdx; j++) {
        for(int j=0; j<outIdx-1; j++) {
          T* cand_ptr = const_cast<T*>(&dataset((size_t)next_cand.idx, 0));

 //         if (new_nbh_list[j].idx != raft::upper_bound<IdxT>()) {
            T* mem_ptr = const_cast<T*>(&dataset((size_t)new_nbh_list[j].idx, 0));
            accT dist_starprime = dist<T, accT>(cand_ptr, mem_ptr, dim, metric);
            if(threadIdx.x==0) graphDist = dist_starprime;
            __syncthreads();

//            if (alpha * dist_starprime <= new_nbh_list[j].dist) {
//            if (alpha * graphDist <= new_nbh_list[j].dist) {
            if (alpha * graphDist <= next_cand.dist) {
              good = false;
              j = outIdx;
            }
//          }
        }
      }
      if(good) {
        new_nbh_list[outIdx].idx = next_cand.idx;
        new_nbh_list[outIdx].dist = next_cand.dist;
        outIdx++;
      }
    }
    __syncthreads();
    // Copy results out to graph
    for(int j=threadIdx.x; j<degree; j+=blockDim.x) {
     query_list[i].ids[j] = new_nbh_list[j].idx;
     query_list[i].dists[j] = new_nbh_list[j].dist;
    }
    if(threadIdx.x==0) {
      query_list[i].size = res_size;
    }
  }
}

}  // namespace

}  // namespace cuvs::neighbors::vamana::detail
