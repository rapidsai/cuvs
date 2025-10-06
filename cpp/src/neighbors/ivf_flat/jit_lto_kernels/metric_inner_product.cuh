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
struct inner_prod_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    if constexpr (Veclen > 1 && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
      acc = raft::dp4a(x, y, acc);
    } else {
      acc += x * y;
    }
  }
};

template <int Veclen, typename T, typename AccT>
__device__ void compute_dist(AccT& acc, AccT x, AccT y)
{
  inner_prod_dist<Veclen, T, AccT>{}(acc, x, y);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
