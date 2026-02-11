/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_dev_essentials.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>

namespace cuvs::distance::detail::sparse {
__device__ __inline__ auto get_lowest_peer(unsigned int peer_group) -> unsigned int
{
  return __ffs(peer_group) - 1;
}

/**
 * Load-balanced sparse-matrix-sparse-matrix multiplication (SPMM) kernel with
 * sparse-matrix-sparse-vector multiplication layout (SPMV).
 * This is intended to be scheduled n_chunks_b times for each row of a.
 * The steps are as follows:
 *
 * 1. Load row from a into dense vector in shared memory.
 *    This can be further chunked in the future if necessary to support larger
 *    column sizes.
 * 2. Threads of block all step through chunks of B in parallel.
 *    When a new row is encountered in row_indices_b, a segmented
 *    reduction is performed across the warps and then across the
 *    block and the final value written out to host memory.
 *
 * Reference: https://www.icl.utk.edu/files/publications/2020/icl-utk-1421-2020.pdf
 *
 * @tparam ValueIdx index type
 * @tparam ValueT value type
 * @tparam tpb threads per block configured on launch
 * @tparam rev if this is true, the reduce/accumulate functions are only
 *         executed when a[col] == 0.0. when executed before/after !rev
 *         and a & B are reversed, this allows the full symmetric difference
 *         and intersection to be computed.
 * @tparam kv_t data type stored in shared mem cache
 * @tparam ProductF reduce function type (semiring product() function).
 *                  accepts two arguments of ValueT and returns a ValueT
 * @tparam AccumF accumulation function type (semiring sum() function).
 *                 accepts two arguments of ValueT and returns a ValueT
 * @tparam WriteF function to write value out. this should be mathematically
 *                 equivalent to the accumulate function but implemented as
 *                 an atomic operation on global memory. Accepts two arguments
 *                 of ValueT* and ValueT and updates the value given by the
 *                 pointer.
 * @param[in] indptrA column pointer array for a
 * @param[in] indicesA column indices array for a
 * @param[in] dataA data array for a
 * @param[in] rowsB coo row array for B
 * @param[in] indicesB column indices array for B
 * @param[in] dataB data array for B
 * @param[in] m number of rows in a
 * @param[in] n number of rows in B
 * @param[in] dim number of features
 * @param[in] nnz_b number of nonzeros in B
 * @param[out] out array of size m*n
 * @param[in] n_blocks_per_row number of blocks of B per row of a
 * @param[in] chunk_size number of nnz for B to use for each row of a
 * @param[in] buffer_size amount of smem to use for each row of a
 * @param[in] product_func semiring product() function
 * @param[in] accum_func semiring sum() function
 * @param[in] write_func atomic semiring sum() function
 */
template <typename StrategyT,
          typename IndptrIt,
          typename ValueIdx,
          typename ValueT,  // NOLINT(readability-identifier-naming)
          bool rev,
          int tpb,
          typename ProductF,
          typename AccumF,
          typename WriteF>
RAFT_KERNEL balanced_coo_generalized_spmv_kernel(StrategyT strategy,
                                                 IndptrIt indptrA,
                                                 ValueIdx* indicesA,
                                                 ValueT* dataA,
                                                 ValueIdx nnz_a,
                                                 ValueIdx* rowsB,
                                                 ValueIdx* indicesB,
                                                 ValueT* dataB,
                                                 ValueIdx m,
                                                 ValueIdx n,
                                                 int dim,
                                                 ValueIdx nnz_b,
                                                 ValueT* out,
                                                 int n_blocks_per_row,
                                                 int chunk_size,
                                                 ValueIdx b_ncols,
                                                 ProductF product_func,
                                                 AccumF accum_func,
                                                 WriteF write_func)
{
  using warp_reduce = cub::WarpReduce<ValueT>;

  ValueIdx cur_row_a        = indptrA.get_row_idx(n_blocks_per_row);
  ValueIdx cur_chunk_offset = blockIdx.x % n_blocks_per_row;

  // chunk starting offset
  ValueIdx ind_offset = cur_chunk_offset * chunk_size * tpb;
  // how many total cols will be processed by this block (should be <= chunk_size * n_threads)
  ValueIdx active_chunk_size = min(chunk_size * tpb, nnz_b - ind_offset);

  int tid     = threadIdx.x;
  int warp_id = tid / raft::warp_size();

  // compute id relative to current warp
  unsigned int lane_id = tid & (raft::warp_size() - 1);
  ValueIdx ind         = ind_offset + threadIdx.x;

  extern __shared__ char smem[];

  void* a = smem;
  typename warp_reduce::TempStorage* temp_storage =
    (typename warp_reduce::TempStorage*)((char*)a + dim);

  auto map_ref = strategy.init_map(a, dim);

  __syncthreads();

  ValueIdx start_offset_a, stop_offset_a;
  bool first_a_chunk, last_a_chunk;
  indptrA.get_row_offsets(
    cur_row_a, start_offset_a, stop_offset_a, n_blocks_per_row, first_a_chunk, last_a_chunk);

  // Convert current row vector in a to dense
  for (int i = tid; i <= (stop_offset_a - start_offset_a); i += blockDim.x) {
    strategy.insert(map_ref, indicesA[start_offset_a + i], dataA[start_offset_a + i]);
  }

  __syncthreads();

  if (cur_row_a > m || cur_chunk_offset > n_blocks_per_row) return;
  if (ind >= nnz_b) return;

  ValueIdx start_index_a = 0, stop_index_a = b_ncols - 1;
  indptrA.get_indices_boundary(indicesA,
                               cur_row_a,
                               start_offset_a,
                               stop_offset_a,
                               start_index_a,
                               stop_index_a,
                               first_a_chunk,
                               last_a_chunk);

  ValueIdx cur_row_b = -1;
  ValueT c           = 0.0;

  auto warp_red = warp_reduce(*(temp_storage + warp_id));

  if (tid < active_chunk_size) {
    cur_row_b = rowsB[ind];

    auto index_b   = indicesB[ind];
    auto in_bounds = indptrA.check_indices_bounds(start_index_a, stop_index_a, index_b);

    if (in_bounds) {
      ValueT a_col = strategy.find(map_ref, index_b);
      if (!rev || a_col == 0.0) { c = product_func(a_col, dataB[ind]); }
    }
  }

  // loop through chunks in parallel, reducing when a new row is
  // encountered by each thread
  for (int i = tid; i < active_chunk_size; i += blockDim.x) {
    ValueIdx ind_next   = ind + blockDim.x;
    ValueIdx next_row_b = -1;

    if (i + blockDim.x < active_chunk_size) next_row_b = rowsB[ind_next];

    bool diff_rows = next_row_b != cur_row_b;

    if (__any_sync(0xffffffff, diff_rows)) {
      // grab the threads currently participating in loops.
      // because any other threads should have returned already.
      unsigned int peer_group = __match_any_sync(0xffffffff, cur_row_b);
      bool is_leader          = get_lowest_peer(peer_group) == lane_id;
      ValueT v                = warp_red.HeadSegmentedReduce(c, is_leader, accum_func);

      // thread with lowest lane id among peers writes out
      if (is_leader && v != 0.0) {
        // this conditional should be uniform, since rev is constant
        size_t idx = !rev ? (size_t)cur_row_a * n + cur_row_b : (size_t)cur_row_b * m + cur_row_a;
        write_func(out + idx, v);
      }

      c = 0.0;
    }

    if (next_row_b != -1) {
      ind = ind_next;

      auto index_b   = indicesB[ind];
      auto in_bounds = indptrA.check_indices_bounds(start_index_a, stop_index_a, index_b);
      if (in_bounds) {
        ValueT a_col = strategy.find(map_ref, index_b);

        if (!rev || a_col == 0.0) { c = accum_func(c, product_func(a_col, dataB[ind])); }
      }

      cur_row_b = next_row_b;
    }
  }
}

}  // namespace cuvs::distance::detail::sparse
