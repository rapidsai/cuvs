/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

/**
 * @brief the L1 distance matrix calculation
 *
 * It computes the following equation:
 *
 *   c_ij = sum_k abs(x_ik  - y_kj)
 */
template <typename DataType, typename AccType, typename IdxType>
struct l1_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  // Do not load norms of data, the computation of L1 distance does not use them.
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

  DI auto core(acc_t& acc, data_t& x, data_t& y) const -> void
  {
    acc += raft::abs(raft::to_float(x) - raft::to_float(y));
  };

  template <typename Policy>
  DI auto epilog(acc_t acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const -> void
  {
    return;
  };
};

}  // namespace cuvs::distance::detail::ops
