/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/quantize/scalar.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>

namespace cuvs::preprocessing::quantize::detail {

template <class T>
_RAFT_HOST_DEVICE auto fp_lt(const T& a, const T& b) -> bool
{
  return a < b;
}

template <>
_RAFT_HOST_DEVICE auto fp_lt(const half& a, const half& b) -> bool
{
  return static_cast<float>(a) < static_cast<float>(b);
}

template <typename T, typename QuantI, typename TempT = double>
struct quantize_op {
  const T min_;
  const T max_;
  const QuantI q_type_min = std::numeric_limits<QuantI>::min();
  const QuantI q_type_max = std::numeric_limits<QuantI>::max();
  const TempT scalar;
  const TempT offset;

  constexpr explicit quantize_op(T min, T max)
    : min_(min),
      max_(max),
      scalar(static_cast<TempT>(max_) > static_cast<TempT>(min_)
               ? ((static_cast<TempT>(q_type_max) - static_cast<TempT>(q_type_min)) /
                  (static_cast<TempT>(max_) - static_cast<TempT>(min_)))
               : static_cast<TempT>(1)),
      offset(static_cast<TempT>(q_type_min) - static_cast<TempT>(min_) * scalar)
  {
  }

  constexpr RAFT_INLINE_FUNCTION auto operator()(const T& x) const -> QuantI
  {
    if (!fp_lt(min_, x)) return q_type_min;
    if (!fp_lt(x, max_)) return q_type_max;
    return static_cast<QuantI>(lroundf(scalar * static_cast<TempT>(x) + offset));
  }

  constexpr RAFT_INLINE_FUNCTION auto operator()(const QuantI& x) const -> T
  {
    return static_cast<T>((static_cast<TempT>(x) - offset) / scalar);
  }
};

template <typename T, typename IdxT = int64_t, typename Accessor>
auto quantile_min_max(
  raft::resources const& res,
  raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, Accessor> dataset,
  double quantile) -> std::tuple<T, T>
{
  // settings for quantile approximation
  constexpr size_t kMaxNumSamples = 1000000;
  constexpr int kSeed             = 137;

  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  // select subsample
  raft::random::RngState rng(kSeed);
  size_t n_rows        = dataset.extent(0);
  size_t dim           = dataset.extent(1);
  size_t n_sample_rows = std::min<size_t>(std::ceil(kMaxNumSamples / dim), n_rows);

  // select subsample rows (this returns device data for both device and host input)
  auto subset = raft::matrix::sample_rows(res, rng, dataset, static_cast<IdxT>(n_sample_rows));

  // quantile / sort element-wise and pick for now
  size_t subset_size = n_sample_rows * dim;
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
auto train(raft::resources const& res,
           const cuvs::preprocessing::quantize::scalar::params params,
           raft::device_matrix_view<const T, int64_t> dataset)
  -> cuvs::preprocessing::quantize::scalar::quantizer<T>
{
  RAFT_EXPECTS(params.quantile > 0.0 && params.quantile <= 1.0,
               "quantile for scalar quantization needs to be within (0, 1] but is %f",
               params.quantile);

  auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

  RAFT_LOG_DEBUG("quantizer train min=%lf max=%lf.", double(min), double(max));

  return cuvs::preprocessing::quantize::scalar::quantizer<T>{min, max};
}

template <typename T>
auto train(raft::resources const& res,
           const cuvs::preprocessing::quantize::scalar::params params,
           raft::host_matrix_view<const T, int64_t> dataset)
  -> cuvs::preprocessing::quantize::scalar::quantizer<T>
{
  RAFT_EXPECTS(params.quantile > 0.0 && params.quantile <= 1.0,
               "quantile for scalar quantization needs to be within (0, 1] but is %f",
               params.quantile);

  auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

  RAFT_LOG_DEBUG("quantizer train min=%lf max=%lf.", double(min), double(max));

  return cuvs::preprocessing::quantize::scalar::quantizer<T>{min, max};
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  raft::linalg::map(res, out, quantize_op<T, QuantI>(quantizer.min_, quantizer.max_), dataset);
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
               raft::host_matrix_view<const T, int64_t> dataset,
               raft::host_matrix_view<QuantI, int64_t> out)
{
  auto main_op      = quantize_op<T, QuantI>(quantizer.min_, quantizer.max_);
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }
}

template <typename T, typename QuantI = int8_t>
void inverse_transform(raft::resources const& res,
                       const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
                       raft::device_matrix_view<const QuantI, int64_t> dataset,
                       raft::device_matrix_view<T, int64_t> out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  raft::linalg::map(res, out, quantize_op<T, QuantI>(quantizer.min_, quantizer.max_), dataset);
}

template <typename T, typename QuantI = int8_t>
void inverse_transform(raft::resources const& res,
                       const cuvs::preprocessing::quantize::scalar::quantizer<T>& quantizer,
                       raft::host_matrix_view<const QuantI, int64_t> dataset,
                       raft::host_matrix_view<T, int64_t> out)
{
  auto main_op      = quantize_op<T, QuantI>(quantizer.min_, quantizer.max_);
  size_t n_elements = dataset.extent(0) * dataset.extent(1);

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    out.data_handle()[i] = main_op(dataset.data_handle()[i]);
  }
}

}  // namespace cuvs::preprocessing::quantize::detail
