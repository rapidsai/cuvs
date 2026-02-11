/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

/**
 * @brief the Hamming Unexpanded distance matrix calculation
 *  It computes the following equation:
 *
 *    c_ij = sum_k (x_ik != y_kj) / k
 */
template <typename DataType, typename AccType, typename IdxType>
struct hamming_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  idx_t k;

  explicit hamming_distance_op(idx_t k_) noexcept : k(k_) {}

  // Load norms of input data
  static constexpr bool kUseNorms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool kExpensiveInnerLoop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr auto shared_mem_size() -> size_t
  {
    return Policy::SmemSize;
  }

  DI void core(acc_t& acc, data_t& x, data_t& y) const { acc += (x != y); };

  template <typename Policy>
  DI void epilog(acc_t acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const
  {
    const acc_t one_over_k = acc_t(1.0) / k;
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        acc[i][j] *= one_over_k;
      }
    }
  }
};

}  // namespace cuvs::distance::detail::ops
