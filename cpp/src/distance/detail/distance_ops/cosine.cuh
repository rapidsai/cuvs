/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

// Epilogue operator for CUTLASS based kernel
template <typename DataT, typename AccT>
struct cosine_cutlass_op {
  __device__ cosine_cutlass_op() noexcept = default;
  __device__ auto operator()(AccT& aNorm, const AccT& bNorm, AccT& accVal) const noexcept -> AccT
  {
    return static_cast<AccT>(1.0) - static_cast<AccT>(accVal / (aNorm * bNorm));
  }
  __device__ auto operator()(DataT aData) const noexcept -> AccT { return raft::to_float(aData); }
};

/**
 * @brief the expanded cosine distance matrix calculation
 *
 * It computes the following equation:
 *
 * d(x, y) = 1 - (x ⋅ y) / ( ||x||_2 ||y||_2)
 */
template <typename DataType, typename AccType, typename IdxType>
struct cosine_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

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
    return Policy::SmemSize + ((Policy::Mblk + Policy::Nblk) * sizeof(acc_t));
  }

  DI void core(acc_t& acc, data_t& x, data_t& y) const
  {
    if constexpr ((std::is_same_v<acc_t, float> && std::is_same_v<data_t, half>)) {
      acc += __half2float(x) * __half2float(y);
    } else {
      acc += x * y;
    }
  };

  template <typename Policy>
  DI void epilog(acc_t acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const
  {
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        if constexpr ((std::is_same_v<acc_t, float> && std::is_same_v<acc_t, half>)) {
          acc[i][j] = 1.0 - (acc[i][j] / (__half2float(regxn[i]) * __half2float(regyn[j])));
        } else {
          acc[i][j] = 1.0 - (acc[i][j] / (regxn[i] * regyn[j]));
        }
      }
    }
  }

  constexpr auto get_cutlass_op() const -> cosine_cutlass_op<data_t, acc_t>
  {
    return cosine_cutlass_op<data_t, acc_t>();
  }
};

}  // namespace cuvs::distance::detail::ops
