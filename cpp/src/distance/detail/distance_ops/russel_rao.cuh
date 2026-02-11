/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

/**
 * @brief the Russell Rao distance matrix calculation
 *
 * It computes the following equation:
 *
 *  c_ij = (k - (sum_k x_ik * y_kj)) / k
 */
template <typename DataType, typename AccType, typename IdxType>
struct russel_rao_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  idx_t k;
  const float one_over_k;

  explicit russel_rao_distance_op(idx_t k_) noexcept : k(k_), one_over_k(1.0f / k_) {}

  // Load norms of input data
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
    acc += raft::to_float(x) * raft::to_float(y);
  };

  template <typename policy>
  DI auto epilog(acc_t acc[policy::AccRowsPerTh][policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const -> void
  {
#pragma unroll
    for (int i = 0; i < policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < policy::AccColsPerTh; ++j) {
        acc[i][j] = (k - acc[i][j]) * one_over_k;
      }
    }
  }
};

}  // namespace cuvs::distance::detail::ops
