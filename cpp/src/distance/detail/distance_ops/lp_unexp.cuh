/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/operators.hpp>            // raft::pow, raft::abs
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

/**
 * @brief the unexpanded Lp (Minkowski) distance matrix calculation
 *
 * It computes the following equation:
 *
 *   c_ij = (sum_k |x_ik - y_jk|^p)^(1/p)
 */
template <typename DataType, typename AccType, typename IdxType>
struct lp_unexp_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  data_t p;

  explicit lp_unexp_distance_op(data_t p_) noexcept : p(p_) {}

  // Load norms of input data
  static constexpr bool kUseNorms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool kExpensiveInnerLoop = true;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename policy>
  static constexpr auto shared_mem_size() -> size_t
  {
    return policy::SmemSize;
  }

  DI auto core(acc_t& acc, data_t& x, data_t& y) const -> void
  {
    const acc_t diff = raft::abs(raft::to_float(x) - raft::to_float(y));
    acc += raft::pow(diff, raft::to_float(p));
  };

  template <typename policy>
  DI auto epilog(acc_t acc[policy::AccRowsPerTh][policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const -> void
  {
    const acc_t one_over_p = 1.0f / static_cast<acc_t>(raft::to_float(p));
#pragma unroll
    for (int i = 0; i < policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < policy::AccColsPerTh; ++j) {
        acc[i][j] = raft::pow(acc[i][j], one_over_p);
      }
    }
  }
};

}  // namespace cuvs::distance::detail::ops
