/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/math.hpp>

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

namespace cuvs::distance::detail::sparse {

/**
 * Computes the maximum number of columns that can be stored
 * in shared memory in dense form with the given block size
 * and precision.
 * @return the maximum number of columns that can be stored in smem
 */
template <typename ValueIdx, typename ValueT, int kTpb = 1024>
inline auto max_cols_per_block() -> int
{
  // max cols = (total smem available - cub reduction smem)
  return (raft::getSharedMemPerBlock() - ((kTpb / raft::warp_size()) * sizeof(ValueT))) /
         sizeof(ValueT);
}

template <typename ValueIdx, typename ValueT, typename DotT = ValueT>
RAFT_KERNEL faster_dot_on_csr_kernel(DotT* __restrict__ dot,
                                     const ValueIdx* __restrict__ indptr,
                                     const ValueIdx* __restrict__ cols,
                                     const ValueT* __restrict__ A,
                                     const ValueT* __restrict__ B,
                                     const ValueIdx nnz,
                                     const ValueIdx n_rows,
                                     const ValueIdx dim)
{
  auto vec_id  = threadIdx.x;
  auto lane_id = threadIdx.x & 0x1f;

  extern __shared__ char smem[];
  auto* s_a        = reinterpret_cast<ValueT*>(smem);
  ValueIdx cur_row = -1;

  for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
    for (int dot_id = blockIdx.y + indptr[row]; dot_id < indptr[row + 1]; dot_id += gridDim.y) {
      if (dot_id >= nnz) { return; }
      const ValueIdx col               = cols[dot_id] * dim;
      const ValueT* __restrict__ b_col = B + col;

      if (threadIdx.x == 0) { dot[dot_id] = 0.0; }
      __syncthreads();

      if (cur_row != row) {
        for (ValueIdx k = vec_id; k < dim; k += blockDim.x) {
          s_a[k] = A[row * dim + k];
        }
        cur_row = row;
      }

      DotT l_dot = 0.0;
      for (ValueIdx k = vec_id; k < dim; k += blockDim.x) {
        asm("prefetch.global.L2 [%0];" ::"l"(b_col + k + blockDim.x));
        if constexpr ((std::is_same_v<DotT, float> && std::is_same_v<ValueT, half>)) {
          l_dot += __half2float(s_a[k]) * __half2float(__ldcg(b_col + k));
        } else {
          l_dot += s_a[k] * __ldcg(b_col + k);
        }
      }

      using WarpReduce = cub::WarpReduce<DotT>;
      __shared__ typename WarpReduce::TempStorage temp_storage;
      DotT warp_sum = WarpReduce(temp_storage).Sum(l_dot);

      if (lane_id == 0) { atomicAdd_block(dot + dot_id, warp_sum); }
    }
  }
}

template <typename ValueIdx, typename ValueT, typename DotT = ValueT>
void faster_dot_on_csr(raft::resources const& handle,
                       DotT* dot,
                       const ValueIdx nnz,
                       const ValueIdx* indptr,
                       const ValueIdx* cols,
                       const ValueT* A,
                       const ValueT* B,
                       const ValueIdx n_rows,
                       const ValueIdx dim)
{
  if (nnz == 0 || n_rows == 0) return;

  auto stream = raft::resource::get_cuda_stream(handle);

  constexpr ValueIdx kMaxRowPerIter = 500;
  int dev_id, sm_count, blocks_per_sm;

  const int smem_size = dim * sizeof(ValueT);
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  if (dim < 128) {
    constexpr int kTpb = 64;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>, kTpb, smem_size);
    auto block_x = std::min(n_rows, kMaxRowPerIter);
    auto block_y = (std::min(ValueIdx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>
      <<<blocks, kTpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);

  } else if (dim < 256) {
    constexpr int kTpb = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>, kTpb, smem_size);
    auto block_x = std::min(n_rows, kMaxRowPerIter);
    auto block_y = (std::min(ValueIdx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>
      <<<blocks, kTpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  } else if (dim < 512) {
    constexpr int kTpb = 256;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>, kTpb, smem_size);
    auto block_x = std::min(n_rows, kMaxRowPerIter);
    auto block_y = (std::min(ValueIdx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>
      <<<blocks, kTpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  } else {
    constexpr int kTpb = 512;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>, kTpb, smem_size);
    auto block_x = std::min(n_rows, kMaxRowPerIter);
    auto block_y = (std::min(ValueIdx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<ValueIdx, ValueT, DotT>
      <<<blocks, kTpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  }

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::distance::detail::sparse
