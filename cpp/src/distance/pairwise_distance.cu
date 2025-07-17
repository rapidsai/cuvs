/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "distance.cuh"
#include <cstdint>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>

namespace cuvs::distance {

/**
 * @defgroup pairwise_distance_runtime Pairwise Distances Runtime API
 * @{
 */
void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg)
{
  auto x_v = raft::make_device_matrix_view<const float, int, raft::layout_c_contiguous>(
    x.data_handle(), x.extent(0), x.extent(1));
  auto y_v = raft::make_device_matrix_view<const float, int, raft::layout_c_contiguous>(
    y.data_handle(), y.extent(0), y.extent(1));
  auto d_v = raft::make_device_matrix_view<float, int, raft::layout_c_contiguous>(
    dist.data_handle(), dist.extent(0), dist.extent(1));
  pairwise_distance<float, raft::layout_c_contiguous, int>(
    handle, x_v, y_v, d_v, metric, metric_arg);
}

void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const uint8_t, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const uint8_t, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<uint32_t, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  uint32_t metric_arg)
{
  // Validate that this is BitwiseHamming
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::BitwiseHamming,
               "uint8_t overload only supports BitwiseHamming distance");

  const auto k     = x.extent(1);
  const auto x_ptr = x.data_handle();
  const auto y_ptr = y.data_handle();

  if (k % 8 == 0) {
    // Optimal case: Use uint64_t computation (8x faster)
    auto x_u64 = raft::make_device_matrix_view<const uint64_t, int, raft::layout_c_contiguous>(
      reinterpret_cast<const uint64_t*>(x_ptr), x.extent(0), k / 8);
    auto y_u64 = raft::make_device_matrix_view<const uint64_t, int, raft::layout_c_contiguous>(
      reinterpret_cast<const uint64_t*>(y_ptr), y.extent(0), k / 8);
    auto d_v = raft::make_device_matrix_view<uint32_t, int, raft::layout_c_contiguous>(
      dist.data_handle(), dist.extent(0), dist.extent(1));
    pairwise_distance<uint64_t, raft::layout_c_contiguous, int, uint32_t>(
      handle, x_u64, y_u64, d_v, metric, metric_arg);
  } else if (k % 4 == 0) {
    auto x_u32 = raft::make_device_matrix_view<const uint32_t, int, raft::layout_c_contiguous>(
      reinterpret_cast<const uint32_t*>(x_ptr), x.extent(0), k / 4);
    auto y_u32 = raft::make_device_matrix_view<const uint32_t, int, raft::layout_c_contiguous>(
      reinterpret_cast<const uint32_t*>(y_ptr), y.extent(0), k / 4);
    auto d_v = raft::make_device_matrix_view<uint32_t, int, raft::layout_c_contiguous>(
      dist.data_handle(), dist.extent(0), dist.extent(1));
    pairwise_distance<uint32_t, raft::layout_c_contiguous, int, uint32_t>(
      handle, x_u32, y_u32, d_v, metric, metric_arg);
  } else {
    auto x_v = raft::make_device_matrix_view<const uint8_t, int, raft::layout_c_contiguous>(
      x.data_handle(), x.extent(0), x.extent(1));
    auto y_v = raft::make_device_matrix_view<const uint8_t, int, raft::layout_c_contiguous>(
      y.data_handle(), y.extent(0), y.extent(1));
    auto d_v = raft::make_device_matrix_view<uint32_t, int, raft::layout_c_contiguous>(
      dist.data_handle(), dist.extent(0), dist.extent(1));
    pairwise_distance<uint8_t, raft::layout_c_contiguous, int, uint32_t>(
      handle, x_v, y_v, d_v, metric, metric_arg);
  }
}

void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<double, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  double metric_arg)
{
  auto x_v = raft::make_device_matrix_view<const double, int, raft::layout_c_contiguous>(
    x.data_handle(), x.extent(0), x.extent(1));
  auto y_v = raft::make_device_matrix_view<const double, int, raft::layout_c_contiguous>(
    y.data_handle(), y.extent(0), y.extent(1));
  auto d_v = raft::make_device_matrix_view<double, int, raft::layout_c_contiguous>(
    dist.data_handle(), dist.extent(0), dist.extent(1));
  pairwise_distance<double, raft::layout_c_contiguous, int>(
    handle, x_v, y_v, d_v, metric, metric_arg);
}

void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const x,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_c_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_c_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg)
{
  auto x_v = raft::make_device_matrix_view<const half, int, raft::layout_c_contiguous>(
    x.data_handle(), x.extent(0), x.extent(1));
  auto y_v = raft::make_device_matrix_view<const half, int, raft::layout_c_contiguous>(
    y.data_handle(), y.extent(0), y.extent(1));
  auto d_v = raft::make_device_matrix_view<float, int, raft::layout_c_contiguous>(
    dist.data_handle(), dist.extent(0), dist.extent(1));
  pairwise_distance<half, raft::layout_c_contiguous, int, float>(
    handle, x_v, y_v, d_v, metric, metric_arg);
}

void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const float, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg)
{
  auto x_v = raft::make_device_matrix_view<const float, int, raft::layout_f_contiguous>(
    x.data_handle(), x.extent(0), x.extent(1));
  auto y_v = raft::make_device_matrix_view<const float, int, raft::layout_f_contiguous>(
    y.data_handle(), y.extent(0), y.extent(1));
  auto d_v = raft::make_device_matrix_view<float, int, raft::layout_f_contiguous>(
    dist.data_handle(), dist.extent(0), dist.extent(1));
  pairwise_distance<float, raft::layout_f_contiguous, int>(
    handle, x_v, y_v, d_v, metric, metric_arg);
}

void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const double, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<double, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  double metric_arg)
{
  auto x_v = raft::make_device_matrix_view<const double, int, raft::layout_f_contiguous>(
    x.data_handle(), x.extent(0), x.extent(1));
  auto y_v = raft::make_device_matrix_view<const double, int, raft::layout_f_contiguous>(
    y.data_handle(), y.extent(0), y.extent(1));
  auto d_v = raft::make_device_matrix_view<double, int, raft::layout_f_contiguous>(
    dist.data_handle(), dist.extent(0), dist.extent(1));
  pairwise_distance<double, raft::layout_f_contiguous, int>(
    handle, x_v, y_v, d_v, metric, metric_arg);
}

void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const half, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<float, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  float metric_arg)
{
  auto x_v = raft::make_device_matrix_view<const half, int, raft::layout_f_contiguous>(
    x.data_handle(), x.extent(0), x.extent(1));
  auto y_v = raft::make_device_matrix_view<const half, int, raft::layout_f_contiguous>(
    y.data_handle(), y.extent(0), y.extent(1));
  auto d_v = raft::make_device_matrix_view<float, int, raft::layout_f_contiguous>(
    dist.data_handle(), dist.extent(0), dist.extent(1));
  pairwise_distance<half, raft::layout_f_contiguous, int, float>(
    handle, x_v, y_v, d_v, metric, metric_arg);
}

void pairwise_distance(
  raft::resources const& handle,
  raft::device_matrix_view<const uint8_t, std::int64_t, raft::layout_f_contiguous> const x,
  raft::device_matrix_view<const uint8_t, std::int64_t, raft::layout_f_contiguous> const y,
  raft::device_matrix_view<uint32_t, std::int64_t, raft::layout_f_contiguous> dist,
  cuvs::distance::DistanceType metric,
  uint32_t metric_arg)
{
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::BitwiseHamming,
               "uint8_t overload only supports BitwiseHamming distance");

  const auto k     = x.extent(1);
  const auto x_ptr = x.data_handle();
  const auto y_ptr = y.data_handle();

  if (k % 8 == 0) {
    auto x_u64 = raft::make_device_matrix_view<const uint64_t, int, raft::layout_f_contiguous>(
      reinterpret_cast<const uint64_t*>(x_ptr), x.extent(0), k / 8);
    auto y_u64 = raft::make_device_matrix_view<const uint64_t, int, raft::layout_f_contiguous>(
      reinterpret_cast<const uint64_t*>(y_ptr), y.extent(0), k / 8);
    auto d_v = raft::make_device_matrix_view<uint64_t, int, raft::layout_f_contiguous>(
      reinterpret_cast<uint64_t*>(dist.data_handle()), dist.extent(0), dist.extent(1));
    pairwise_distance<uint64_t, raft::layout_f_contiguous, int, uint32_t>(
      handle, x_u64, y_u64, d_v);
  } else if (k % 4 == 0) {
    auto x_u32 = raft::make_device_matrix_view<const uint32_t, int, raft::layout_f_contiguous>(
      reinterpret_cast<const uint32_t*>(x_ptr), x.extent(0), k / 4);
    auto y_u32 = raft::make_device_matrix_view<const uint32_t, int, raft::layout_f_contiguous>(
      reinterpret_cast<const uint32_t*>(y_ptr), y.extent(0), k / 4);
    auto d_v = raft::make_device_matrix_view<uint32_t, int, raft::layout_f_contiguous>(
      dist.data_handle(), dist.extent(0), dist.extent(1));
    pairwise_distance<uint32_t, raft::layout_f_contiguous, int, uint32_t>(
      handle, x_u32, y_u32, d_v);
  } else {
    // Fallback case: Use uint8_t computation (compatible with any alignment)
    auto x_v = raft::make_device_matrix_view<const uint8_t, int, raft::layout_f_contiguous>(
      x.data_handle(), x.extent(0), x.extent(1));
    auto y_v = raft::make_device_matrix_view<const uint8_t, int, raft::layout_f_contiguous>(
      y.data_handle(), y.extent(0), y.extent(1));
    auto d_v = raft::make_device_matrix_view<uint32_t, int, raft::layout_f_contiguous>(
      dist.data_handle(), dist.extent(0), dist.extent(1));
    pairwise_distance<uint8_t, raft::layout_f_contiguous, int, uint32_t>(handle, x_v, y_v, d_v);
  }
}

/** @} */  // end group pairwise_distance_runtime

}  // namespace cuvs::distance
