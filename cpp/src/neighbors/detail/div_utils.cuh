/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/util/pow2_utils.cuh>

/**
 * @brief A simple wrapper for raft::Pow2 which uses Pow2 utils only when available and regular
 * integer division otherwise. This is done to allow a common interface for division arithmetic for
 * non CUDA headers.
 *
 * @tparam Value_ a compile-time value representable as a power-of-two.
 */
namespace cuvs::neighbors::detail {
template <auto Value_>
struct div_utils {
  typedef decltype(Value_) Type;
  static constexpr Type Value = Value_;

  template <typename T>
  static constexpr _RAFT_HOST_DEVICE inline auto roundDown(T x)
  {
    return raft::Pow2<Value_>::roundDown(x);
  }

  template <typename T>
  static constexpr _RAFT_HOST_DEVICE inline auto mod(T x)
  {
    return raft::Pow2<Value_>::mod(x);
  }

  template <typename T>
  static constexpr _RAFT_HOST_DEVICE inline auto div(T x)
  {
    return raft::Pow2<Value_>::div(x);
  }
};
}  // namespace cuvs::neighbors::detail
