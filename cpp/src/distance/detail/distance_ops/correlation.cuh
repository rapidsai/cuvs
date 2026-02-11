/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

/** @brief The correlation distance
 *
 * It computes the following equation:
 *
 * d(x, y) = ((x - mean(x)) ⋅ (y - mean(y)))
 *           /
 *           (|| x - mean(x) ||_2 || y - mean(y) ||_2)
 */
template <typename DataType, typename AccType, typename IdxType>
struct correlation_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  const acc_t* x2n;
  const acc_t* y2n;
  idx_t m;
  idx_t n;
  idx_t k;

  correlation_distance_op(
    bool is_row_major, const acc_t* x2n_, const acc_t* y2n_, idx_t m_, idx_t n_, idx_t k_) noexcept
    : x2n(x2n_), y2n(y2n_), m(m_), n(n_), k(k_)
  {
    // The distance op is typically created before the row-major/col-major
    // swapping has been done. So we do it here.
    if (!is_row_major) {
      std::swap<const acc_t*>(x2n, y2n);
      std::swap(m, n);
    }
  }

  // Load norms of input data
  static constexpr bool kUseNorms = true;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool kExpensiveInnerLoop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr auto shared_mem_size() -> size_t
  {
    return Policy::SmemSize + (2 * (Policy::Mblk + Policy::Nblk) * sizeof(acc_t));
  }

  DI void core(acc_t& acc, data_t& x, data_t& y) const
  {
    acc += raft::to_float(x) * raft::to_float(y);
  };

  template <typename Policy>
  DI void epilog(acc_t acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const
  {
    // Note how we can sneakily get a pointer to shared memory here, to store
    // more data. If the implementation of PairwiseDistanceMatKernel ever
    // changes, this will be where we find the bugs.
    extern __shared__ char smem[];

    acc_t regx2n[Policy::AccRowsPerTh], regy2n[Policy::AccColsPerTh];

    auto* sx2_norm = reinterpret_cast<acc_t*>(
      &smem[Policy::SmemSize + (Policy::Mblk + Policy::Nblk) * sizeof(acc_t)]);
    acc_t* sy2_norm = (&sx2_norm[Policy::Mblk]);

    // Load x & y norms required by this threadblock in shmem buffer
    if (gridStrideX == blockIdx.x * Policy::Nblk) {
      for (int i = threadIdx.x; i < Policy::Mblk; i += Policy::Nthreads) {
        auto idx    = gridStrideY + i;
        sx2_norm[i] = idx < m ? raft::to_float(x2n[idx]) : 0;
      }
    }

    for (int i = threadIdx.x; i < Policy::Nblk; i += Policy::Nthreads) {
      auto idx    = gridStrideX + i;
      sy2_norm[i] = idx < n ? raft::to_float(y2n[idx]) : 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
      regx2n[i] = sx2_norm[i * Policy::AccThRows + (threadIdx.x / Policy::AccThCols)];
    }
#pragma unroll
    for (int i = 0; i < Policy::AccColsPerTh; ++i) {
      regy2n[i] = sy2_norm[i * Policy::AccThCols + (threadIdx.x % Policy::AccThCols)];
    }

#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        auto numer   = k * acc[i][j] - (regxn[i] * regyn[j]);
        auto q_denom = k * regx2n[i] - (regxn[i] * regxn[i]);
        auto r_denom = k * regy2n[j] - (regyn[j] * regyn[j]);

        acc[i][j] = 1 - (numer / raft::sqrt(q_denom * r_denom));
      }
    }
  }
};

}  // namespace cuvs::distance::detail::ops
