/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

/**
 * @brief the unexpanded euclidean distance matrix calculation
 *
 * It computes the following equation:
 *
 * c_ij = optional_sqrt ( sum_k (x_ik - y_kj)^2 )
 */
template <typename DataType, typename AccType, typename IdxType>
struct l2_unexp_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  bool sqrt;

  explicit l2_unexp_distance_op(bool sqrt_) noexcept : sqrt(sqrt_) {}

  // Do not load norms of data, the computation of L1 distance does not use them.
  static constexpr bool kUseNorms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool kExpensiveInnerLoop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename policy>
  static constexpr auto shared_mem_size() -> size_t
  {
    return policy::SmemSize;
  }

  DI auto core(acc_t& acc, data_t& x, data_t& y) const -> void
  {
    if constexpr ((std::is_same_v<acc_t, float> && std::is_same_v<data_t, half>)) {
      const auto diff = __half2float(x) - __half2float(y);
      acc += diff * diff;
    } else {
      const auto diff = x - y;
      acc += diff * diff;
    }
  };

  template <typename policy>
  DI auto epilog(acc_t acc[policy::AccRowsPerTh][policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const -> void
  {
    if (sqrt) {
#pragma unroll
      for (int i = 0; i < policy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < policy::AccColsPerTh; ++j) {
          acc[i][j] = raft::sqrt(acc[i][j]);
        }
      }
    }
  };
};

}  // namespace cuvs::distance::detail::ops
