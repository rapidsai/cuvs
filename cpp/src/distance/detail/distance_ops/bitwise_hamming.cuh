/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace cuvs::distance::detail::ops {

/**
 * @brief the Bitwise Hamming distance matrix calculation
 *  It computes the following equation:
 *
 *    c_ij = sum_k popcount(x_ik XOR y_kj)
 *
 * where x and y are binary data packed as uint8_t
 */
template <typename DataType, typename AccType, typename IdxType>
struct bitwise_hamming_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  IdxT k;

  bitwise_hamming_distance_op(IdxT k_) noexcept : k(k_) {}

  static constexpr bool use_norms            = false;
  static constexpr bool expensive_inner_loop = false;

  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  __device__ __forceinline__ void core(AccT& acc, DataT& x, DataT& y) const
  {
    static_assert(std::is_same_v<DataT, uint8_t>, "BitwiseHamming only supports uint8_t");
    // Ensure proper masking and casting to avoid undefined behavior
    uint32_t xor_val    = static_cast<uint32_t>(static_cast<uint8_t>(x ^ y));
    uint32_t masked_val = xor_val & 0xffu;
    int popcount        = __popc(masked_val);
    acc += static_cast<AccT>(popcount);
  }

  template <typename Policy>
  __device__ __forceinline__ void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                                         AccT* regxn,
                                         AccT* regyn,
                                         IdxT gridStrideX,
                                         IdxT gridStrideY) const
  {
  }
};

}  // namespace cuvs::distance::detail::ops
