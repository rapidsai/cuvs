/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <thrust/sort.h>

#include <raft/util/cuda_dev_essentials.cuh>

#include <type_traits>

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
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
__global__ void RobustPruneKernel(
  raft::device_matrix_view<IdxT, int64_t> graph,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
  void* query_list_ptr,
  int num_queries,
  int visited_size,
  cuvs::distance::DistanceType metric,
  float alpha,
  typename greedy_search_query_coord<T>::type* s_coords_mem)
{
  int n      = dataset.extent(0);
  int dim    = dataset.extent(1);
  int degree = graph.extent(1);
  QueryCandidates<IdxT, accT>* query_list =
    static_cast<QueryCandidates<IdxT, accT>*>(query_list_ptr);

  using QueryCoordT = typename greedy_search_query_coord<T>::type;

  union ShmemLayout {
    // All blocksort sizes have same alignment (16)
    float occlusion;
    DistPair<IdxT, accT> nbh_list;
  };

  // Dynamic shared memory used for blocksort, temp vector storage, and neighborhood list
  extern __shared__ __align__(alignof(ShmemLayout)) char smem[];

  int align_padding = raft::alignTo<int>(dim, alignof(ShmemLayout)) - dim;

  float* occlusion_list = reinterpret_cast<float*>(smem);
  const int nbh_list_offset = (degree + visited_size) * sizeof(float);
  DistPair<IdxT, accT>* new_nbh_list =
    reinterpret_cast<DistPair<IdxT, accT>*>(&smem[nbh_list_offset]);
  QueryCoordT* s_cand_coords = nullptr;
  if (dim >= kRobustPruneCandCacheMinDim) {
    s_cand_coords = reinterpret_cast<QueryCoordT*>(
      &smem[nbh_list_offset + (degree + visited_size) * sizeof(DistPair<IdxT, accT>)]);
  }

  static __shared__ Point<QueryCoordT, accT> s_query;
  s_query.coords = &s_coords_mem[blockIdx.x * (dim + align_padding)];
  s_query.Dim    = dim;
  static __shared__ int prev_edges;
  static __shared__ accT graphDist;
  static __shared__ int s_accept_count;
  static __shared__ int s_do_accept;

  const int laneId   = threadIdx.x & 31;
  const int warpId   = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;

  for (int i = blockIdx.x; i < num_queries; i += gridDim.x) {
    int queryId = query_list[i].queryId;

    if constexpr (is_cuda_fp16_v<T>) {
      update_shared_point_half_to_float<accT>(&s_query, &dataset(0, 0), queryId, dim);
    } else {
      update_shared_point<T, accT>(&s_query, &dataset(0, 0), queryId, dim, i);
    }

    int graphIdx = 0;
    int listIdx  = 0;
    int res_size = degree + visited_size;

    // Count total valid edge candidates
    __syncthreads();
    if (threadIdx.x == 0) {
      prev_edges = degree;
      for (int j = 0; j < degree; j++) {
        if (graph(queryId, j) == raft::upper_bound<IdxT>()) {
          prev_edges = j;
          break;
        }
      }
    }
    for (int j = threadIdx.x; j < degree + visited_size; j += blockDim.x) {
      occlusion_list[j] = 0.0;
    }
    __syncthreads();

    DistPair<IdxT, accT> next_cand;
    // Merge graph and candidate list
    for (int outIdx = 0; outIdx < degree + visited_size; outIdx++) {
      // Check if no more valid elements from graph or list
      if (graphIdx < degree && graph(queryId, graphIdx) == raft::upper_bound<IdxT>()) {
        graphIdx = degree;
      }
      if (listIdx < visited_size && query_list[i].ids[listIdx] == raft::upper_bound<IdxT>()) {
        listIdx = visited_size;
      }

      // Get next candidate vector for list
      if (graphIdx >= degree) {
        if (listIdx >= visited_size) {               // Fill remaining list if no candidates
          if (res_size > outIdx) res_size = outIdx;  // Set result size
          new_nbh_list[outIdx].idx  = raft::upper_bound<IdxT>();
          new_nbh_list[outIdx].dist = raft::upper_bound<accT>();
          __syncthreads();
          continue;
        } else {
          next_cand.idx  = query_list[i].ids[listIdx];
          next_cand.dist = query_list[i].dists[listIdx];
          listIdx++;
        }
      } else if (listIdx >= visited_size) {
        next_cand.idx = graph(queryId, graphIdx);
        if (warpId == 0) {
          accT tempDist;
          if constexpr (is_cuda_fp16_v<T>) {
            tempDist = dist_warp<accT>(s_query.coords,
                                       &dataset((size_t)graph(queryId, graphIdx), 0),
                                       dim,
                                       metric,
                                       laneId);
          } else {
            tempDist = dist_warp<T, accT>(s_query.coords,
                                          &dataset((size_t)graph(queryId, graphIdx), 0),
                                          dim,
                                          metric,
                                          laneId);
          }
          if (laneId == 0) graphDist = tempDist;
        }
        __syncthreads();
        next_cand.dist = graphDist;
        graphIdx++;
      } else {
        accT listDist = query_list[i].dists[listIdx];

        if (warpId == 0) {
          accT tempDist;
          if constexpr (is_cuda_fp16_v<T>) {
            tempDist = dist_warp<accT>(s_query.coords,
                                       &dataset((size_t)graph(queryId, graphIdx), 0),
                                       dim,
                                       metric,
                                       laneId);
          } else {
            tempDist = dist_warp<T, accT>(s_query.coords,
                                          &dataset((size_t)graph(queryId, graphIdx), 0),
                                          dim,
                                          metric,
                                          laneId);
          }
          if (laneId == 0) graphDist = tempDist;
        }
        __syncthreads();

        if (listDist <= graphDist) {
          next_cand.idx  = query_list[i].ids[listIdx];
          next_cand.dist = listDist;

          if (graph(queryId, graphIdx) == query_list[i].ids[listIdx]) {  // Duplicate found!
            graphIdx++;                                                  // Skip the duplicate
          }
          listIdx++;
        } else {
          next_cand.idx  = graph(queryId, graphIdx);
          next_cand.dist = graphDist;
          graphIdx++;
        }
      }

      new_nbh_list[outIdx].idx  = next_cand.idx;
      new_nbh_list[outIdx].dist = next_cand.dist;
    }

    // If we need to prune at all...
    if (res_size > degree) {
      if (threadIdx.x == 0) s_accept_count = 0;
      __syncthreads();
      const bool cache_cand_in_smem = dim >= kRobustPruneCandCacheMinDim;
      Point<QueryCoordT, accT> s_cand;
      if (cache_cand_in_smem) {
        s_cand.coords = s_cand_coords;
        s_cand.Dim    = dim;
      }

      // Go through different alpha values. These constants are hard-coded in the MSFT DiskANN code
      for (float cur_alpha = 1.0; cur_alpha <= alpha && s_accept_count < degree;
           cur_alpha *= 1.2) {
        for (int pass_start = 0; pass_start < res_size && s_accept_count < degree; pass_start++) {
          if (threadIdx.x == 0) {
            s_do_accept = (occlusion_list[pass_start] != raft::lower_bound<float>() &&
                           occlusion_list[pass_start] <= cur_alpha &&
                           new_nbh_list[pass_start].idx != queryId)
                            ? 1
                            : 0;
          }
          __syncthreads();

          if (s_do_accept && cache_cand_in_smem) {
            if constexpr (is_cuda_fp16_v<T>) {
              update_shared_point_half_to_float<accT>(
                &s_cand, &dataset(0, 0), new_nbh_list[pass_start].idx, dim);
            } else {
              update_shared_point<T, accT>(
                &s_cand, &dataset(0, 0), new_nbh_list[pass_start].idx, dim);
            }
          }
          __syncthreads();

          if (s_do_accept) {
            if (threadIdx.x == 0) {
              occlusion_list[pass_start] = raft::lower_bound<float>();
              s_accept_count++;
            }

            T* cand_ptr = const_cast<T*>(&dataset((size_t)(new_nbh_list[pass_start].idx), 0));
            for (int occId = pass_start + 1 + warpId; occId < res_size; occId += num_warps) {
              if (occlusion_list[occId] <= alpha &&
                  occlusion_list[occId] != raft::lower_bound<float>()) {
                T* k_ptr = const_cast<T*>(&dataset((size_t)(new_nbh_list[occId].idx), 0));
                accT djk;
                if (cache_cand_in_smem) {
                  if constexpr (is_cuda_fp16_v<T>) {
                    djk = dist_warp<accT>(s_cand.coords, k_ptr, dim, metric, laneId);
                  } else {
                    djk = dist_warp<QueryCoordT, accT>(s_cand.coords, k_ptr, dim, metric, laneId);
                  }
                } else {
                  djk = dist_warp<T, accT>(cand_ptr, k_ptr, dim, metric, laneId);
                }
                if (laneId == 0) {
                  accT new_occ = (float)(new_nbh_list[occId].dist / djk);
                  occlusion_list[occId] = std::max(occlusion_list[occId], new_occ);
                }
              }
            }
          }
          __syncthreads();
        }
      }

      // Move all "accepted" candidates to front of list and zero out the rest
      if (threadIdx.x == 0) {
        int out_idx = 1;
        for (int read_idx = 1; out_idx < s_accept_count; read_idx++) {
          if (occlusion_list[read_idx] == raft::lower_bound<float>()) {  // If it is "accepted"
            new_nbh_list[out_idx].idx  = new_nbh_list[read_idx].idx;
            new_nbh_list[out_idx].dist = new_nbh_list[read_idx].dist;
            out_idx++;
          }
        }
      }
      __syncthreads();
      for (int out_idx = s_accept_count + threadIdx.x; out_idx < degree; out_idx += blockDim.x) {
        new_nbh_list[out_idx].idx  = raft::upper_bound<IdxT>();
        new_nbh_list[out_idx].dist = raft::upper_bound<accT>();
      }

      if (threadIdx.x == 0) { res_size = s_accept_count; }
      __syncthreads();
    }

    // Copy results out to graph
    for (int j = threadIdx.x; j < degree; j += blockDim.x) {
      query_list[i].ids[j]   = new_nbh_list[j].idx;
      query_list[i].dists[j] = new_nbh_list[j].dist;
    }
    if (threadIdx.x == 0) { query_list[i].size = res_size; }
  }
}

}  // namespace

}  // namespace cuvs::neighbors::vamana::detail
