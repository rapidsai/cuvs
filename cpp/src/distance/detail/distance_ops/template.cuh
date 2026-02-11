/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

// Describes the computation the template distance
//
// Fill in the TODO items.

template <typename DataType, typename AccType, typename IdxType>
struct template_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  TODO member;

  explicit template_distance_op(TODO member_) noexcept : member(member_) {}

  // Load norms of input data
  static constexpr bool kUseNorms = TODO;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool kExpensiveInnerLoop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename policy>
  static constexpr auto shared_mem_size() -> size_t
  {
    return policy::SmemSize + TODO;
  }

  DI auto core(acc_t& acc, data_t& x, data_t& y) const -> void { TODO; };

  template <typename policy>
  DI auto epilog(acc_t acc[policy::AccRowsPerTh][policy::AccColsPerTh],
                 acc_t* regxn,
                 acc_t* regyn,
                 idx_t gridStrideX,
                 idx_t gridStrideY) const -> void
  {
    TODO;
  }

  // If exist, returns a cutlass op that performs the same operation.
  // See cosine and l2_exp distance ops for an example.
  [[nodiscard]] constexpr auto get_cutlass_op() const -> l2_exp_cutlass_op<data_t, acc_t> { TODO; }
};

}  // namespace cuvs::distance::detail::ops
