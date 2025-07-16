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

#include <cstdint>
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace cuvs::distance::detail::ops {

/**
 * @brief the BitwiseHamming distance matrix calculation
 *
 * It computes the following equation:
 *
 * c_ij = sum_k popcount(x_ik XOR y_kj)
 *
 * This counts the number of differing bits between corresponding elements
 * across all dimensions.
 */
template <typename DataType, typename AccType, typename IdxType>
struct bitwise_hamming_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  bitwise_hamming_distance_op() noexcept {}

  // Load norms of input data
  static constexpr bool use_norms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    if constexpr (sizeof(DataT) == 1) {
      acc += __popc(static_cast<uint32_t>(x ^ y) & 0xffu);
    } else if constexpr (sizeof(DataT) == 2) {
      acc += __popc(static_cast<uint32_t>(x ^ y) & 0xffffu);
    } else if constexpr (sizeof(DataT) == 4) {
      acc += __popc(x ^ y);
    } else if constexpr (sizeof(DataT) == 8) {
      acc += __popcll(x ^ y);
    } else {
      static_assert(sizeof(DataT) <= 8,
                    "BitwiseHamming distance only supports types up to 64 bits");
    }
  };

  template <typename Policy>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 AccT* regxn,
                 AccT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    // No normalization needed for bitwise Hamming distance
    // The result is the raw count of differing bits
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        // acc[i][j] already contains the correct bitwise Hamming distance
        // No additional processing needed
      }
    }
  }
};

}  // namespace cuvs::distance::detail::ops
