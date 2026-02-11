/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>            // raft::abs
#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

/**
 * @brief The canberra distance matrix calculation
 *
 * It computes the following equation:
 *
 *  c_ij = sum_k |x_ik - y_kj| / ( |x_ik| + |y_kj| )
 */
template <typename DataType, typename AccType, typename IdxType>
struct canberra_distance_op {
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
    if constexpr ((std::is_same_v<acc_t, float> && std::is_same_v<data_t, half>)) {
      acc_t hx        = __half2float(x);
      acc_t hy        = __half2float(y);
      const auto diff = raft::abs(hx - hy);
      const auto add  = raft::abs(hx) + raft::abs(hy);
      // deal with potential for 0 in denominator by
      // forcing 0/1 instead
      acc += ((add != 0) * diff / (add + (add == 0)));
    } else {
      const auto diff = raft::abs(x - y);
      const auto add  = raft::abs(x) + raft::abs(y);
      // deal with potential for 0 in denominator by
      // forcing 0/1 instead
      acc += ((add != 0) * diff / (add + (add == 0)));
    }
  };

  template <typename Policy>
  DI void epilog(acc_t acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const
  {
    return;
  }
};

}  // namespace cuvs::distance::detail::ops
