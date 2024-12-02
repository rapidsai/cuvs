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

#pragma once

#include <cuvs/neighbors/quantization.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>

namespace cuvs::neighbors::detail {

template <typename T, typename QuantI, typename TempT = double>
struct quantize_op {
  const T min_;
  const T max_;
  const bool inverse_;
  const QuantI q_type_min_ = std::numeric_limits<QuantI>::min();
  const QuantI q_type_max_ = std::numeric_limits<QuantI>::max();
  const TempT a_;
  const TempT b_;

  constexpr explicit quantize_op(T min, T max, bool inverse)
    : min_(min),
      max_(max),
      inverse_(inverse),
      a_(static_cast<TempT>(max_) > static_cast<TempT>(min_)
           ? ((static_cast<TempT>(q_type_max_) - static_cast<TempT>(q_type_min_)) /
              (static_cast<TempT>(max_) - static_cast<TempT>(min_)))
           : static_cast<TempT>(1)),
      b_(static_cast<TempT>(q_type_min_) - static_cast<TempT>(min_) * a_)
  {
  }

  constexpr RAFT_INLINE_FUNCTION QuantI operator()(const T& x) const
  {
    if (x > max_) return q_type_max_;
    if (x < min_) return q_type_min_;
    return static_cast<QuantI>(lroundf(a_ * static_cast<TempT>(x) + b_));
  }

  constexpr RAFT_INLINE_FUNCTION T operator()(const QuantI& x) const
  {
    return static_cast<T>((static_cast<TempT>(x) - b_) / a_);
  }
};

template <typename T>
std::tuple<T, T> quantile_min_max(raft::resources const& res,
                                  raft::device_matrix_view<const T, int64_t> dataset,
                                  double quantile)
{
  // settings for quantile approximation
  constexpr size_t max_num_samples = 1000000;
  constexpr int seed               = 137;

  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // select subsample
  raft::random::RngState rng(seed);
  size_t n_elements  = dataset.extent(0) * dataset.extent(1);
  size_t subset_size = std::min(max_num_samples, n_elements);
  auto subset        = raft::make_device_vector<T>(res, subset_size);
  auto dataset_view  = raft::make_device_vector_view<const T>(dataset.data_handle(), n_elements);
  raft::random::sample_without_replacement(
    res, rng, dataset_view, std::nullopt, subset.view(), std::nullopt);

  // quantile / sort and pick for now
  thrust::sort(raft::resource::get_thrust_policy(res),
               subset.data_handle(),
               subset.data_handle() + subset_size);

  double half_quantile_pos = (0.5 + 0.5 * quantile) * subset_size;
  int pos_max              = std::ceil(half_quantile_pos) - 1;
  int pos_min              = subset_size - pos_max - 1;

  T minmax_h[2];
  raft::update_host(&(minmax_h[0]), subset.data_handle() + pos_min, 1, stream);
  raft::update_host(&(minmax_h[1]), subset.data_handle() + pos_max, 1, stream);
  raft::resource::sync_stream(res);

  return {minmax_h[0], minmax_h[1]};
}

template <typename T>
std::tuple<T, T> quantile_min_max(raft::resources const& res,
                                  raft::host_matrix_view<const T, int64_t> dataset,
                                  double quantile)
{
  // settings for quantile approximation
  constexpr size_t max_num_samples = 1000000;
  constexpr int seed               = 137;

  // select subsample
  std::mt19937 rng(seed);
  size_t n_elements  = dataset.extent(0) * dataset.extent(1);
  size_t subset_size = std::min(max_num_samples, n_elements);
  std::vector<T> subset;
  std::sample(dataset.data_handle(),
              dataset.data_handle() + n_elements,
              std::back_inserter(subset),
              subset_size,
              rng);

  // quantile / sort and pick for now
  thrust::sort(thrust::omp::par, subset.data(), subset.data() + subset_size);
  double half_quantile_pos = (0.5 + 0.5 * quantile) * subset_size;
  int pos_max              = std::ceil(half_quantile_pos) - 1;
  int pos_min              = subset_size - pos_max - 1;

  return {subset[pos_min], subset[pos_max]};
}

template <typename T, typename QuantI = int8_t>
raft::device_matrix<QuantI, int64_t> scalar_transform(
  raft::resources const& res, raft::device_matrix_view<const T, int64_t> dataset, T min, T max)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // allocate target
  auto out = raft::make_device_matrix<QuantI, int64_t>(res, dataset.extent(0), dataset.extent(1));

  raft::linalg::map(res, out.view(), quantize_op<T, QuantI>(min, max, false), dataset);

  return out;
}

template <typename T, typename QuantI = int8_t>
raft::host_matrix<QuantI, int64_t> scalar_transform(
  raft::resources const& res, raft::host_matrix_view<const T, int64_t> dataset, T min, T max)
{
  // allocate target
  auto out = raft::make_host_matrix<QuantI, int64_t>(dataset.extent(0), dataset.extent(1));

  auto main_op      = quantize_op<T, QuantI>(min, max, false);
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }

  return out;
}

template <typename T, typename QuantI = int8_t>
raft::device_matrix<T, int64_t> inverse_scalar_transform(
  raft::resources const& res, raft::device_matrix_view<const QuantI, int64_t> dataset, T min, T max)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // allocate target
  auto out = raft::make_device_matrix<T, int64_t>(res, dataset.extent(0), dataset.extent(1));

  raft::linalg::map(res, out.view(), quantize_op<T, QuantI>(min, max, true), dataset);

  return out;
}

template <typename T, typename QuantI = int8_t>
raft::host_matrix<T, int64_t> inverse_scalar_transform(
  raft::resources const& res, raft::host_matrix_view<const QuantI, int64_t> dataset, T min, T max)
{
  // allocate target
  auto out = raft::make_host_matrix<T, int64_t>(dataset.extent(0), dataset.extent(1));

  auto main_op      = quantize_op<T, QuantI>(min, max, true);
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }

  return out;
}

}  // namespace cuvs::neighbors::detail
