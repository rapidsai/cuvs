/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/math.hpp>
#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

namespace cuvs::distance::detail::ops {

/**
 * Reserve 1 digit of precision from each floating-point type
 * for round-off error tolerance.
 * @tparam DataT
 */
template <typename DataT, typename AccT>
__device__ constexpr auto get_clamp_precision() -> AccT
{
  switch (sizeof(DataT)) {
    case 2: return AccT{1e-3};
    case 4: return AccT{1e-6};
    case 8: return AccT{1e-15};
    default: return AccT{0};
  }
}

// Epilogue operator for CUTLASS based kernel
template <typename DataT, typename AccT>
struct l2_exp_cutlass_op {
  bool sqrt;

  __device__ l2_exp_cutlass_op() noexcept : sqrt(false) {}
  __device__ explicit l2_exp_cutlass_op(bool isSqrt) noexcept : sqrt(isSqrt) {}
  inline __device__ auto operator()(AccT aNorm, AccT bNorm, AccT accVal) const noexcept -> AccT
  {
    AccT out_val = aNorm + bNorm - AccT(2.0) * accVal;

    /**
     * Self-neighboring points should have (aNorm == bNorm) == accVal and the dot product (accVal)
     * can sometimes have round-off errors, which will cause (aNorm == bNorm) ~ accVal instead.
     */
    out_val = out_val *
              AccT(!((out_val * out_val < get_clamp_precision<DataT, AccT>()) * (aNorm == bNorm)));
    return sqrt ? raft::sqrt(out_val * static_cast<AccT>(out_val > AccT(0))) : out_val;
  }

  __device__ auto operator()(DataT aData) const noexcept -> AccT
  {
    if constexpr (std::is_same_v<DataT, half> && std::is_same_v<AccT, float>) {
      return __half2float(aData);
    } else {
      return aData;
    }
  }
};

/**
 * @brief the expanded euclidean distance matrix calculation
 *
 * It computes the following equation:
 *
 * c_ij = - 2 sum_k x_ik * y_kj + ||x_i.||_2 + ||y_.j||_2
 *
 */
template <typename DataType, typename AccType, typename IdxType>
struct l2_exp_distance_op {
  using data_t = DataType;
  using acc_t  = AccType;
  using idx_t  = IdxType;

  const bool sqrt;

  explicit l2_exp_distance_op(bool sqrt_) noexcept : sqrt(sqrt_) {}

  // Load norms of input data
  static constexpr bool kUseNorms = true;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool kExpensiveInnerLoop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename policy>
  static constexpr auto shared_mem_size() -> size_t
  {
    return policy::SmemSize + ((policy::Mblk + policy::Nblk) * sizeof(acc_t));
  }

  DI auto core(acc_t& acc, data_t& x, data_t& y) const -> void
  {
    if constexpr ((std::is_same_v<acc_t, float> && std::is_same_v<data_t, half>)) {
      acc += __half2float(x) * __half2float(y);
    } else {
      acc += x * y;
    }
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
        acc_t acc_val = acc[i][j];
        acc_t val     = regxn[i] + regyn[j] - static_cast<acc_t>(2.0) * acc_val;

        /**
         * Self-neighboring points should have (aNorm == bNorm) == accVal and the dot product
         * (accVal) can sometimes have round-off errors, which will cause (aNorm == bNorm) ~ accVal
         * instead.
         */
        acc[i][j] = val * static_cast<acc_t>((val > acc_t(0))) *
                    static_cast<acc_t>(!((val * val < get_clamp_precision<data_t, acc_t>()) *
                                         (regxn[i] == regyn[j])));
      }
    }
    if (sqrt) {
#pragma unroll
      for (int i = 0; i < policy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < policy::AccColsPerTh; ++j) {
          acc[i][j] = raft::sqrt(acc[i][j]);
        }
      }
    }
  }

  constexpr auto get_cutlass_op() const -> l2_exp_cutlass_op<data_t, acc_t>
  {
    return l2_exp_cutlass_op<data_t, acc_t>(sqrt);
  }
};

}  // namespace cuvs::distance::detail::ops
