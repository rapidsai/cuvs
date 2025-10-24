/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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

/** @} */  // end group pairwise_distance_runtime

}  // namespace cuvs::distance
