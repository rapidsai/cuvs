/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <optional>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::all_neighbors::detail {

template <typename IdxT, int BLOCK_SIZE, int ITEMS_PER_THREAD, bool SweepAll = false>
RAFT_KERNEL merge_subgraphs_kernel(IdxT* cluster_data_indices,
                                   size_t graph_degree,
                                   size_t num_cluster_in_batch,
                                   float* global_distances,
                                   float* batch_distances,
                                   IdxT* global_indices,
                                   IdxT* batch_indices,
                                   bool select_min)
{
  size_t batch_row = blockIdx.x;
  using BlockMergeSortType =
    cub::BlockMergeSort<raft::KeyValuePair<float, IdxT>, BLOCK_SIZE, ITEMS_PER_THREAD>;
  __shared__ typename cub::BlockMergeSort<raft::KeyValuePair<float, IdxT>,
                                          BLOCK_SIZE,
                                          ITEMS_PER_THREAD>::TempStorage tmpSmem;

  extern __shared__ char shared_mem[];
  auto* block_keys   = reinterpret_cast<float*>(shared_mem);
  IdxT* block_values = reinterpret_cast<IdxT*>(&shared_mem[graph_degree * 2 * sizeof(float)]);
  auto* unique_mask =
    reinterpret_cast<int16_t*>(&shared_mem[graph_degree * 2 * (sizeof(float) + sizeof(IdxT))]);

  if (batch_row < num_cluster_in_batch) {
    // load batch or global depending on threadIdx
    size_t global_row = cluster_data_indices[batch_row];

    raft::KeyValuePair<float, IdxT> thread_key_value_pair[ITEMS_PER_THREAD];

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

    size_t idx_base = (threadIdx.x * do_global + (threadIdx.x - halfway) * (1lu - do_global)) *
                      static_cast<size_t>(ITEMS_PER_THREAD);
    size_t arr_idx_base = (global_row * do_global + batch_row * (1lu - do_global)) * graph_degree;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t col_id = idx_base + i;
      if (col_id < graph_degree) {
        thread_key_value_pair[i].key   = distances[arr_idx_base + col_id];
        thread_key_value_pair[i].value = indices[arr_idx_base + col_id];
      } else {
        if (select_min) {
          thread_key_value_pair[i].key   = std::numeric_limits<float>::max();
          thread_key_value_pair[i].value = std::numeric_limits<IdxT>::max();
        } else {
          thread_key_value_pair[i].key   = std::numeric_limits<float>::min();
          thread_key_value_pair[i].value = std::numeric_limits<IdxT>::min();
        }
      }
    }

    __syncthreads();

    if (select_min) {
      BlockMergeSortType(tmpSmem).Sort(thread_key_value_pair, raft::less_op{});
    } else {
      BlockMergeSortType(tmpSmem).Sort(thread_key_value_pair, raft::greater_op{});
    }

    __syncthreads();

    size_t limit = 2 * graph_degree;
    // load sorted result into shared memory to get unique values
    idx_base = threadIdx.x * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t col_id = idx_base + i;
      if (col_id < limit) {
        block_keys[col_id]   = thread_key_value_pair[i].key;
        block_values[col_id] = thread_key_value_pair[i].value;
      }
    }

    __syncthreads();

    // get unique mask
    if (threadIdx.x == 0) { unique_mask[0] = 1; }
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t col_id = idx_base + i;
      if (col_id > 0 && col_id < limit) {
        // this assumes same distance between vector from two different batches. however, currently
        // there are subtle differences in the result based on the matrix size used to call gemm.
        // This makes it difficult to remove duplicates, because they might no longer be right next
        // to each other after sorting by distances. Thus, for now we sweep a neighboring window of
        // size 4 or sweep the entire row to check for duplicates, and keep the first occurrence
        // only.
        // related issue: https://github.com/rapidsai/cuvs/issues/1056
        // unique_mask[col_id] = static_cast<int16_t>(block_values[col_id] != block_values[col_id -
        // 1]);

        int is_unique = 1;

        if constexpr (SweepAll) {  // sweep whole row for better deduplication
          for (int j = 0; j < limit; ++j) {
            if (j < col_id && block_values[j] == block_values[col_id]) {
              is_unique = 0;
              break;
            }
          }
        } else {  // otherwise sweep a small window
          IdxT curr_val = block_values[col_id];
#pragma unroll
          for (int offset = -4; offset < 0; offset++) {
            int neighbor_idx = static_cast<int>(col_id) + offset;
            if (neighbor_idx >= 0) {
              if (block_values[neighbor_idx] == curr_val) {
                is_unique = 0;
                break;
              }
            }
          }
        }
        unique_mask[col_id] = static_cast<int16_t>(is_unique);
      }
    }

    __syncthreads();

    // prefix sum
    if (threadIdx.x == 0) {
      for (int i = 1; i < limit; i++) {
        unique_mask[i] += unique_mask[i - 1];
      }
    }

    __syncthreads();
    // load unique values to global memory
    if (threadIdx.x == 0) {
      global_distances[global_row * graph_degree] = block_keys[0];
      global_indices[global_row * graph_degree]   = block_values[0];
    }

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      size_t col_id = idx_base + i;
      if (col_id > 0 && col_id < limit) {
        bool is_unique        = unique_mask[col_id] != unique_mask[col_id - 1];
        int16_t global_col_id = unique_mask[col_id] - 1;
        if (is_unique && static_cast<size_t>(global_col_id) < graph_degree) {
          global_distances[global_row * graph_degree + global_col_id] = block_keys[col_id];
          global_indices[global_row * graph_degree + global_col_id]   = block_values[col_id];
        }
      }
    }
  }
}

template <typename T, typename IdxT = int64_t, bool SweepAll = false>
void merge_subgraphs(raft::resources const& res,
                     size_t k,
                     size_t num_data_in_cluster,
                     IdxT* inverted_indices_d,
                     T* global_distances,
                     T* batch_distances_d,
                     IdxT* global_neighbors,
                     IdxT* batch_neighbors_d,
                     bool select_min)
{
  size_t num_elems       = k * 2;
  size_t shared_mem_size = num_elems * (sizeof(float) + sizeof(IdxT) + sizeof(int16_t));

#pragma omp critical  // for omp-using multi-gpu purposes
  {
    if (num_elems <= 128) {
      merge_subgraphs_kernel<IdxT, 32, 4, SweepAll>
        <<<num_data_in_cluster, 32, shared_mem_size, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d,
          select_min);
    } else if (num_elems <= 512) {
      merge_subgraphs_kernel<IdxT, 128, 4, SweepAll>
        <<<num_data_in_cluster, 128, shared_mem_size, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d,
          select_min);
    } else if (num_elems <= 1024) {
      merge_subgraphs_kernel<IdxT, 128, 8, SweepAll>
        <<<num_data_in_cluster, 128, shared_mem_size, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d,
          select_min);
    } else if (num_elems <= 2048) {
      merge_subgraphs_kernel<IdxT, 256, 8, SweepAll>
        <<<num_data_in_cluster, 256, shared_mem_size, raft::resource::get_cuda_stream(res)>>>(
          inverted_indices_d,
          k,
          num_data_in_cluster,
          global_distances,
          batch_distances_d,
          global_neighbors,
          batch_neighbors_d,
          select_min);
    } else {
      // this is as far as we can get due to the shared mem usage of cub::BlockMergeSort
      RAFT_FAIL("The degree of knn is too large (%lu). It must be smaller than 1024", k);
    }
    raft::resource::sync_stream(res);
  }
}

template <typename T,
          typename IdxT         = int64_t,
          typename BeforeRemapT = int64_t,
          bool SweepAll         = false>
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
                               size_t k,
                               bool select_min)
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

  merge_subgraphs<T, IdxT, SweepAll>(res,
                                     k,
                                     num_data_in_cluster,
                                     inverted_indices_d.data_handle(),
                                     global_distances.data_handle(),
                                     batch_distances_d.data_handle(),
                                     global_neighbors.data_handle(),
                                     batch_neighbors_d.data_handle(),
                                     select_min);
}

}  // namespace cuvs::neighbors::all_neighbors::detail
