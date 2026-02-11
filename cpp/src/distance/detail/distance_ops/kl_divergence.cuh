/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/operators.hpp>            // raft::log
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

/**
 * @brief the KL Divergence distance matrix calculation
 *
 * It computes the following equation:
 *
 *   c_ij = 0.5 * sum(x * log (x / y));
 */
template <typename DataType, typename AccType, typename IdxType>
struct kl_divergence_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  const bool is_row_major;
  const bool x_equal_y;

  explicit kl_divergence_op(bool row_major_, bool x_equal_y_ = false) noexcept
    : is_row_major(row_major_), x_equal_y(x_equal_y_)
  {
  }

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

  DI auto core(acc_t& acc, data_t& x, data_t& y) const -> void
  {
    // TODO(snanditale): make sure that these branches get hoisted out of main loop.. Could
    // be quite expensive otherwise.
    acc_t xv = raft::to_float(x);
    acc_t yv = raft::to_float(y);
    if (x_equal_y) {
      if (is_row_major) {
        const bool x_zero = (xv == 0);
        const bool y_zero = (yv == 0);
        acc += xv * (raft::log(xv + x_zero) - (!y_zero) * raft::log(yv + y_zero));
      } else {
        const bool y_zero = (yv == 0);
        const bool x_zero = (xv == 0);
        acc += yv * (raft::log(yv + y_zero) - (!x_zero) * raft::log(xv + x_zero));
      }
    } else {
      if (is_row_major) {
        const bool x_zero = (xv == 0);
        acc += xv * (raft::log(xv + x_zero) - yv);
      } else {
        const bool y_zero = (yv == 0);
        acc += yv * (raft::log(yv + y_zero) - xv);
      }
    }
  };

  template <typename Policy>
  DI auto epilog(acc_t acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const -> void
  {
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        acc[i][j] = (0.5f * acc[i][j]);
      }
    }
  }
};
}  // namespace cuvs::distance::detail::ops
