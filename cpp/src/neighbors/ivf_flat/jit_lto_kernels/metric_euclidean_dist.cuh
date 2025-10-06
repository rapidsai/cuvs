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

#include "interleaved_scan_tags.hpp"
#include <raft/util/cuda_utils.cuh>

namespace cuvs::neighbors::ivf_flat::detail {

template <int Veclen, typename T, typename AccT>
struct euclidean_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    const auto diff = x - y;
    acc += diff * diff;
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, uint8_t, uint32_t> {
  __device__ __forceinline__ void operator()(uint32_t& acc, uint32_t x, uint32_t y)
  {
    if constexpr (Veclen > 1) {
      const auto diff = __vabsdiffu4(x, y);
      acc             = raft::dp4a(diff, diff, acc);
    } else {
      const auto diff = __usad(x, y, 0u);
      acc += diff * diff;
    }
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, int8_t, int32_t> {
  __device__ __forceinline__ void operator()(int32_t& acc, int32_t x, int32_t y)
  {
    if constexpr (Veclen > 1) {
      // Note that we enforce here that the unsigned version of dp4a is used, because the difference
      // between two int8 numbers can be greater than 127 and therefore represented as a negative
      // number in int8. Casting from int8 to int32 would yield incorrect results, while casting
      // from uint8 to uint32 is correct.
      const auto diff = __vabsdiffs4(x, y);
      acc             = raft::dp4a(diff, diff, static_cast<uint32_t>(acc));
    } else {
      const auto diff = x - y;
      acc += diff * diff;
    }
  }
};

template <int Veclen, typename T, typename AccT>
__device__ void compute_dist(AccT& acc, AccT x, AccT y)
{
  euclidean_dist<Veclen, T, AccT>{}(acc, x, y);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
