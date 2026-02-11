/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "faiss_select/Select.cuh"
#include <raft/core/error.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <cstdint>

namespace cuvs::neighbors::detail {

template <typename ValueIdx = std::int64_t,
          typename ValueT   = float,
          int warp_q,
          int thread_q,
          int tpb>
RAFT_KERNEL knn_merge_parts_kernel(const ValueT* inK,
                                   const ValueIdx* inV,
                                   ValueT* outK,
                                   ValueIdx* outV,
                                   size_t n_samples,
                                   int n_parts,
                                   ValueT initK,
                                   ValueIdx initV,
                                   int k,
                                   ValueIdx* translations)
{
  constexpr int kNumWarps = tpb / raft::WarpSize;

  __shared__ ValueT smem_k[kNumWarps * warp_q];
  __shared__ ValueIdx smem_v[kNumWarps * warp_q];

  /**
   * Uses shared memory
   */
  cuvs::neighbors::detail::faiss_select::block_select<
    ValueT,
    ValueIdx,
    false,
    cuvs::neighbors::detail::faiss_select::comparator<ValueT>,
    warp_q,
    thread_q,
    tpb>
    heap(initK, initV, smem_k, smem_v, k);

  // Grid is exactly sized to rows available
  int row     = blockIdx.x;
  int total_k = k * n_parts;

  int i = threadIdx.x;

  // Get starting pointers for cols in current thread
  int part       = i / k;
  size_t row_idx = (row * k) + (part * n_samples * k);

  int col = i % k;

  const ValueT* in_k_start   = inK + (row_idx + col);
  const ValueIdx* in_v_start = inV + (row_idx + col);

  int limit            = raft::Pow2<raft::WarpSize>::roundDown(total_k);
  ValueIdx translation = 0;

  for (; i < limit; i += tpb) {
    translation = translations[part];
    heap.add(*in_k_start, (*in_v_start) + translation);

    part    = (i + tpb) / k;
    row_idx = (row * k) + (part * n_samples * k);

    col = (i + tpb) % k;

    in_k_start = inK + (row_idx + col);
    in_v_start = inV + (row_idx + col);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < total_k) {
    translation = translations[part];
    heap.add_thread_q(*in_k_start, (*in_v_start) + translation);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    outK[row * k + i] = smem_k[i];
    outV[row * k + i] = smem_v[i];
  }
}

template <typename ValueIdx = std::int64_t, typename ValueT = float, int warp_q, int thread_q>
inline void knn_merge_parts_impl(const ValueT* inK,
                                 const ValueIdx* inV,
                                 ValueT* outK,
                                 ValueIdx* outV,
                                 size_t n_samples,
                                 int n_parts,
                                 int k,
                                 cudaStream_t stream,
                                 ValueIdx* translations)
{
  auto grid = dim3(n_samples);

  constexpr int kNThreads = (warp_q < 1024) ? 128 : 64;
  auto block              = dim3(kNThreads);

  auto k_init = std::numeric_limits<ValueT>::max();
  auto v_init = -1;
  knn_merge_parts_kernel<ValueIdx, ValueT, warp_q, thread_q, kNThreads><<<grid, block, 0, stream>>>(
    inK, inV, outK, outV, n_samples, n_parts, k_init, v_init, k, translations);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Merge knn distances and index matrix, which have been partitioned
 * by row, into a single matrix with only the k-nearest neighbors.
 *
 * @param inK partitioned knn distance matrix
 * @param inV partitioned knn index matrix
 * @param outK merged knn distance matrix
 * @param outV merged knn index matrix
 * @param n_samples number of samples per partition
 * @param n_parts number of partitions
 * @param k number of neighbors per partition (also number of merged neighbors)
 * @param stream CUDA stream to use
 * @param translations mapping of index offsets for each partition
 */
template <typename ValueIdx = std::int64_t, typename ValueT = float>
inline void knn_merge_parts(const ValueT* inK,
                            const ValueIdx* inV,
                            ValueT* outK,
                            ValueIdx* outV,
                            size_t n_samples,
                            int n_parts,
                            int k,
                            cudaStream_t stream,
                            ValueIdx* translations)
{
  if (k == 1) {
    knn_merge_parts_impl<ValueIdx, ValueT, 1, 1>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  } else if (k <= 32) {
    knn_merge_parts_impl<ValueIdx, ValueT, 32, 2>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  } else if (k <= 64) {
    knn_merge_parts_impl<ValueIdx, ValueT, 64, 3>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  } else if (k <= 128) {
    knn_merge_parts_impl<ValueIdx, ValueT, 128, 3>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  } else if (k <= 256) {
    knn_merge_parts_impl<ValueIdx, ValueT, 256, 4>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  } else if (k <= 512) {
    knn_merge_parts_impl<ValueIdx, ValueT, 512, 8>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  } else if (k <= 1024) {
    knn_merge_parts_impl<ValueIdx, ValueT, 1024, 8>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  } else {
    THROW("Unimplemented for k=%d, knn_merge_parts works for k<=1024", k);
  }
}
}  // namespace cuvs::neighbors::detail
