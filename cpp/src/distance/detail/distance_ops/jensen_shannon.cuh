/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/operators.hpp>            // raft::log
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

// Describes the computation the jensen_shannon distance

/**
 * @brief the Jensen Shannon distance matrix calculation
 *
 * It computes the following equation:
 *
 * c_ij = sqrt(0.5 * sum( -x_i * (log(0.5 * (x_i + y_i)) - log(x_i))
 *       + (-y_i * (log(0.5 * (x_i + y_i)) - log(y_i)))))
 */
template <typename DataType, typename AccType, typename IdxType>
struct jensen_shannon_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  // Load norms of input data
  static constexpr bool kUseNorms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool kExpensiveInnerLoop = true;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr auto shared_mem_size() -> size_t
  {
    return Policy::SmemSize;
  }

  DI void core(acc_t& acc, data_t& x, data_t& y) const
  {
    acc_t xv          = raft::to_float(x);
    acc_t yv          = raft::to_float(y);
    const acc_t m     = 0.5f * (xv + yv);
    const bool m_zero = (m == 0);
    const auto log_m  = (!m_zero) * raft::log(m + m_zero);

    const bool x_zero = (xv == 0);
    const bool y_zero = (yv == 0);
    acc += (-xv * (log_m - raft::log(xv + x_zero))) + (-yv * (log_m - raft::log(yv + y_zero)));
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
        acc[i][j] = raft::sqrt(0.5 * acc[i][j]);
      }
    }
  }
};

}  // namespace cuvs::distance::detail::ops
