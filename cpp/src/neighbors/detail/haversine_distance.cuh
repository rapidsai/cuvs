/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "faiss_select/Select.cuh"
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <cuda_fp16.h>

namespace cuvs::neighbors::detail {
template <typename ValueT, typename DistanceT>  // NOLINT(readability-identifier-naming)
DI auto compute_haversine(ValueT x1, ValueT y1, ValueT x2, ValueT y2) -> DistanceT
{
  if constexpr ((std::is_same_v<DistanceT, float> && std::is_same_v<ValueT, half>)) {
    DistanceT x1_f = __half2float(x1);
    DistanceT y1_f = __half2float(y1);
    DistanceT x2_f = __half2float(x2);
    DistanceT y2_f = __half2float(y2);

    DistanceT sin_0 = raft::sin(DistanceT(0.5) * (x1_f - y1_f));
    DistanceT sin_1 = raft::sin(DistanceT(0.5) * (x2_f - y2_f));
    DistanceT rdist = sin_0 * sin_0 + raft::cos(x1_f) * raft::cos(y1_f) * sin_1 * sin_1;

    return static_cast<DistanceT>(2) * raft::asin(raft::sqrt(rdist));
  } else {
    DistanceT sin_0 = raft::sin(DistanceT(0.5) * (x1 - y1));
    DistanceT sin_1 = raft::sin(DistanceT(0.5) * (x2 - y2));
    DistanceT rdist = sin_0 * sin_0 + raft::cos(x1) * raft::cos(y1) * sin_1 * sin_1;

    return static_cast<DistanceT>(2) * raft::asin(raft::sqrt(rdist));
  }
}

/**
 * @tparam ValueIdx data type of indices
 * @tparam ValueT data type of values and distances
 * @tparam warp_q
 * @tparam thread_q
 * @tparam tpb
 * @param[out] out_inds output indices
 * @param[out] out_dists output distances
 * @param[in] index index array
 * @param[in] query query array
 * @param[in] n_index_rows number of rows in index array
 * @param[in] k number of closest neighbors to return
 */
template <typename ValueIdx,
          typename ValueT,  // NOLINT(readability-identifier-naming)
          int warp_q         = 1024,
          int thread_q       = 8,
          int tpb            = 128,
          typename DistanceT = float>
RAFT_KERNEL haversine_knn_kernel(ValueIdx* out_inds,
                                 DistanceT* out_dists,
                                 const ValueT* index,
                                 const ValueT* query,
                                 size_t n_index_rows,
                                 int k)
{
  constexpr int kNumWarps = tpb / raft::WarpSize;

  __shared__ DistanceT smem_k[kNumWarps * warp_q];
  __shared__ ValueIdx smem_v[kNumWarps * warp_q];

  using cuvs::neighbors::detail::faiss_select::block_select;
  using cuvs::neighbors::detail::faiss_select::comparator;
  block_select<DistanceT, ValueIdx, false, comparator<DistanceT>, warp_q, thread_q, tpb> heap(
    std::numeric_limits<DistanceT>::max(), std::numeric_limits<ValueIdx>::max(), smem_k, smem_v, k);

  // Grid is exactly sized to rows available
  int limit = raft::Pow2<raft::WarpSize>::roundDown(n_index_rows);

  const ValueT* query_ptr = query + (blockIdx.x * 2);
  ValueT x1               = query_ptr[0];
  ValueT x2               = query_ptr[1];

  int i = threadIdx.x;

  for (; i < limit; i += tpb) {
    const ValueT* idx_ptr = index + (i * 2);
    ValueT y1             = idx_ptr[0];
    ValueT y2             = idx_ptr[1];

    DistanceT dist = compute_haversine<ValueT, DistanceT>(x1, y1, x2, y2);

    heap.add(dist, i);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < n_index_rows) {
    const ValueT* idx_ptr = index + (i * 2);
    ValueT y1             = idx_ptr[0];
    ValueT y2             = idx_ptr[1];

    DistanceT dist = compute_haversine<ValueT, DistanceT>(x1, y1, x2, y2);

    heap.add_thread_q(dist, i);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    out_dists[blockIdx.x * k + i] = smem_k[i];
    out_inds[blockIdx.x * k + i]  = smem_v[i];
  }
}

/**
 * Conmpute the k-nearest neighbors using the Haversine
 * (great circle arc) distance. Input is assumed to have
 * 2 dimensions (latitude, longitude) in radians.

 * @tparam ValueIdx
 * @tparam ValueT
 * @param[out] out_inds output indices array on device (size n_query_rows * k)
 * @param[out] out_dists output dists array on device (size n_query_rows * k)
 * @param[in] index input index array on device (size n_index_rows * 2)
 * @param[in] query input query array on device (size n_query_rows * 2)
 * @param[in] n_index_rows number of rows in index array
 * @param[in] n_query_rows number of rows in query array
 * @param[in] k number of closest neighbors to return
 * @param[in] stream stream to order kernel launch
 */
template <typename ValueIdx,
          typename ValueT,
          typename DistanceT = float>  // NOLINT(readability-identifier-naming)
void haversine_knn(ValueIdx* out_inds,
                   DistanceT* out_dists,
                   const ValueT* index,
                   const ValueT* query,
                   size_t n_index_rows,
                   size_t n_query_rows,
                   int k,
                   cudaStream_t stream)
{
  // ensure kernel does not breach shared memory limits
  constexpr int kWarpQ = sizeof(ValueT) > 4 ? 512 : 1024;
  haversine_knn_kernel<ValueIdx, ValueT, kWarpQ>
    <<<n_query_rows, 128, 0, stream>>>(out_inds, out_dists, index, query, n_index_rows, k);
}

}  // namespace cuvs::neighbors::detail
