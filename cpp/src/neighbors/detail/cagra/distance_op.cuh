/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/distance/distance.hpp>

namespace {
template <typename T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(T a, T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::L2Expanded, T>
{
  T diff = a - b;
  return diff * diff;
}

template <typename T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(T a, T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::InnerProduct, T>
{
  return -a * b;
}
}  // namespace
