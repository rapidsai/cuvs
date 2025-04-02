/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::all_neighbors::detail {
using namespace cuvs::neighbors;
using align32 = raft::Pow2<32>;

template <typename KeyType, typename ValueType>
struct CustomComparator {
  __device__ bool operator()(const raft::KeyValuePair<KeyType, ValueType>& a,
                             const raft::KeyValuePair<KeyType, ValueType>& b) const
  {
    return a < b;
  }
};

template <typename IdxT, int BLOCK_SIZE, int ITEMS_PER_THREAD>
RAFT_KERNEL merge_subgraphs_kernel(IdxT* cluster_data_indices,
                                   size_t graph_degree,
                                   size_t num_cluster_in_batch,
                                   float* global_distances,
                                   float* batch_distances,
                                   IdxT* global_indices,
                                   IdxT* batch_indices)
{
  size_t batch_row = blockIdx.x;
  typedef cub::BlockMergeSort<raft::KeyValuePair<float, IdxT>, BLOCK_SIZE, ITEMS_PER_THREAD>
    BlockMergeSortType;
  __shared__ typename cub::BlockMergeSort<raft::KeyValuePair<float, IdxT>,
                                          BLOCK_SIZE,
                                          ITEMS_PER_THREAD>::TempStorage tmpSmem;

  extern __shared__ char sharedMem[];
  float* blockKeys  = reinterpret_cast<float*>(sharedMem);
  IdxT* blockValues = reinterpret_cast<IdxT*>(&sharedMem[graph_degree * 2 * sizeof(float)]);
  int16_t* uniqueMask =
    reinterpret_cast<int16_t*>(&sharedMem[graph_degree * 2 * (sizeof(float) + sizeof(IdxT))]);

  if (batch_row < num_cluster_in_batch) {
    // load batch or global depending on threadIdx
    size_t global_row = cluster_data_indices[batch_row];

    raft::KeyValuePair<float, IdxT> threadKeyValuePair[ITEMS_PER_THREAD];

    size_t halfway   = BLOCK_SIZE / 2;
    size_t do_global = threadIdx.x < halfway;

    float* distances;
    IdxT* indices;

    if (do_global) {
      distances = global_distances;
      indices   = global_indices;
    } else {
      distances = batch_distances;
      indices   = batch_indices;
    }

    size_t idxBase = (threadIdx.x * do_global + (threadIdx.x - halfway) * (1lu - do_global)) *
                     static_cast<size_t>(ITEMS_PER_THREAD);
    size_t arrIdxBase = (global_row * do_global + batch_row * (1lu - do_global)) * graph_degree;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < graph_degree) {
        threadKeyValuePair[i].key   = distances[arrIdxBase + colId];
        threadKeyValuePair[i].value = indices[arrIdxBase + colId];
      } else {
        threadKeyValuePair[i].key   = std::numeric_limits<float>::max();
        threadKeyValuePair[i].value = std::numeric_limits<IdxT>::max();
      }
    }

    __syncthreads();

    BlockMergeSortType(tmpSmem).Sort(threadKeyValuePair, CustomComparator<float, IdxT>{});

    // load sorted result into shared memory to get unique values
    idxBase = threadIdx.x * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId < 2 * graph_degree) {
        blockKeys[colId]   = threadKeyValuePair[i].key;
        blockValues[colId] = threadKeyValuePair[i].value;
      }
    }

    __syncthreads();

    // get unique mask
    if (threadIdx.x == 0) { uniqueMask[0] = 1; }
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        uniqueMask[colId] = static_cast<int16_t>(blockValues[colId] != blockValues[colId - 1]);
      }
    }

    __syncthreads();

    // prefix sum
    if (threadIdx.x == 0) {
      for (int i = 1; i < 2 * graph_degree; i++) {
        uniqueMask[i] += uniqueMask[i - 1];
      }
    }

    __syncthreads();
    // load unique values to global memory
    if (threadIdx.x == 0) {
      global_distances[global_row * graph_degree] = blockKeys[0];
      global_indices[global_row * graph_degree]   = blockValues[0];
    }

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t colId = idxBase + i;
      if (colId > 0 && colId < 2 * graph_degree) {
        bool is_unique       = uniqueMask[colId] != uniqueMask[colId - 1];
        int16_t global_colId = uniqueMask[colId] - 1;
        if (is_unique && static_cast<size_t>(global_colId) < graph_degree) {
          global_distances[global_row * graph_degree + global_colId] = blockKeys[colId];
          global_indices[global_row * graph_degree + global_colId]   = blockValues[colId];
        }
      }
    }
  }
}

template <typename T, typename IdxT = int64_t>
void merge_subgraphs(raft::resources const& res,
                     size_t k,
                     size_t num_data_in_cluster,
                     IdxT* inverted_indices_d,
                     T* global_distances,
                     T* batch_distances_d,
                     IdxT* global_neighbors,
                     IdxT* batch_neighbors_d)
{
  size_t num_elems     = k * 2;
  size_t sharedMemSize = num_elems * (sizeof(float) + sizeof(IdxT) + sizeof(int16_t));

#pragma omp critical  // for omp-using multi-gpu purposes
  {
    if (num_elems <= 128) {
      merge_subgraphs_kernel<IdxT, 32, 4>
        <<<num_data_in_cluster, 32, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else if (num_elems <= 512) {
      merge_subgraphs_kernel<IdxT, 128, 4>
        <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else if (num_elems <= 1024) {
      merge_subgraphs_kernel<IdxT, 128, 8>
        <<<num_data_in_cluster, 128, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else if (num_elems <= 2048) {
      merge_subgraphs_kernel<IdxT, 256, 8>
        <<<num_data_in_cluster, 256, sharedMemSize, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d);
    } else {
      // this is as far as we can get due to the shared mem usage of cub::BlockMergeSort
      RAFT_FAIL("The degree of knn is too large (%lu). It must be smaller than 1024", k);
    }
    raft::resource::sync_stream(res);
  }
}

template <typename T, typename IdxT = int64_t, typename BeforeRemapT = int64_t>
void remap_and_merge_subgraphs(raft::resources const& res,
                               raft::device_vector_view<IdxT, IdxT> inverted_indices_d,
                               raft::host_vector_view<IdxT, IdxT> inverted_indices,
                               raft::host_matrix_view<BeforeRemapT, IdxT> indices_for_remap_h,
                               raft::host_matrix_view<IdxT, IdxT> batch_neighbors_h,
                               raft::device_matrix_view<IdxT, IdxT> batch_neighbors_d,
                               raft::device_matrix_view<T, IdxT> batch_distances_d,
                               raft::managed_matrix_view<IdxT, IdxT> global_neighbors,
                               raft::managed_matrix_view<T, IdxT> global_distances,
                               size_t num_data_in_cluster,
                               size_t k)
{
  // remap indices
#pragma omp parallel for
  for (size_t i = 0; i < num_data_in_cluster; i++) {
    for (size_t j = 0; j < k; j++) {
      size_t local_idx        = indices_for_remap_h(i, j);
      batch_neighbors_h(i, j) = inverted_indices(local_idx);
    }
  }

  raft::copy(inverted_indices_d.data_handle(),
             inverted_indices.data_handle(),
             num_data_in_cluster,
             raft::resource::get_cuda_stream(res));

  raft::copy(batch_neighbors_d.data_handle(),
             batch_neighbors_h.data_handle(),
             num_data_in_cluster * k,
             raft::resource::get_cuda_stream(res));

  merge_subgraphs(res,
                  k,
                  num_data_in_cluster,
                  inverted_indices_d.data_handle(),
                  global_distances.data_handle(),
                  batch_distances_d.data_handle(),
                  global_neighbors.data_handle(),
                  batch_neighbors_d.data_handle());
}

}  // namespace cuvs::neighbors::all_neighbors::detail
