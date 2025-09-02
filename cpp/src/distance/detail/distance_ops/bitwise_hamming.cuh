/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
  static constexpr bool expensive_inner_loop = true;  // Force vec_len=1 to reduce shared memory usage

  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  __device__ __forceinline__ void core(AccT& acc, DataT& x, DataT& y) const
  {
    static_assert(std::is_same_v<DataT, uint8_t>, "BitwiseHamming only supports uint8_t");
    acc += static_cast<AccT>(__popc(static_cast<uint32_t>(x ^ y) & 0xffu));
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
