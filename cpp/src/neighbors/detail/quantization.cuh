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

template <typename T, typename QuantI = int8_t, typename TempT = double, typename TempI = int64_t>
raft::device_matrix<QuantI, int64_t> scalar_quantize(
  raft::resources const& res,
  cuvs::neighbors::quantization::params& params,
  raft::device_matrix_view<const T, int64_t> dataset)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  constexpr TempI q_type_min   = static_cast<TempI>(std::numeric_limits<QuantI>::min());
  constexpr TempI q_type_max   = static_cast<TempI>(std::numeric_limits<QuantI>::max());
  constexpr TempI range_q_type = q_type_max - q_type_min + TempI(1);

  size_t n_elements = dataset.extent(0) * dataset.extent(1);

  // conditional: search for quantiles / min / max
  if (!params.is_computed) {
    ASSERT(params.quantile > 0.5 && params.quantile <= 1.0,
           "quantile for scalar quantization needs to be within (.5, 1]");

    double quantile_inv = 1.0 / params.quantile;

    // select subsample
    int seed                     = 137;
    constexpr size_t num_samples = 10000;
    raft::random::RngState rng(seed);
    size_t subset_size = std::min(num_samples, n_elements);
    auto subset        = raft::make_device_vector<T>(res, subset_size);
    auto dataset_view  = raft::make_device_vector_view<const T>(dataset.data_handle(), n_elements);
    raft::random::sample_without_replacement(
      res, rng, dataset_view, std::nullopt, subset.view(), std::nullopt);

    // quantile / sort and pick for now
    thrust::sort(raft::resource::get_thrust_policy(res),
                 subset.data_handle(),
                 subset.data_handle() + subset_size);

    int pos_max = raft::ceildiv((double)subset_size, quantile_inv) - 1;
    int pos_min = subset_size - pos_max - 1;

    T minmax_h[2];
    raft::update_host(&(minmax_h[0]), subset.data_handle() + pos_min, 1, stream);
    raft::update_host(&(minmax_h[1]), subset.data_handle() + pos_max, 1, stream);
    raft::resource::sync_stream(res);

    // persist settings in params
    params.min         = double(minmax_h[0]);
    params.max         = double(minmax_h[1]);
    params.scalar      = double(range_q_type) / (params.max - params.min + 1.0);
    params.is_computed = true;
  }

  // allocate target
  auto out = raft::make_device_matrix<QuantI, int64_t>(res, dataset.extent(0), dataset.extent(1));

  // raft unary op or raft::linalg::map?
  // TempT / TempI as intermediate types
  raft::linalg::unaryOp(out.data_handle(),
                        dataset.data_handle(),
                        n_elements,
                        raft::compose_op(
                          raft::cast_op<QuantI>{},
                          raft::add_const_op<int>(q_type_min),
                          [] __device__(TempI a) {
                            return raft::max<TempI>(raft::min<TempI>(a, q_type_max - q_type_min),
                                                    TempI(0));
                          },
                          raft::cast_op<TempI>{},
                          raft::add_const_op<TempT>(0.5),  // for rounding
                          raft::mul_const_op<TempT>(params.scalar),
                          raft::sub_const_op<TempT>(params.min),
                          raft::cast_op<TempT>{}),
                        stream);

  return out;
}

template <typename T, typename QuantI = int8_t, typename TempT = double, typename TempI = int64_t>
raft::host_matrix<QuantI, int64_t> scalar_quantize(raft::resources const& res,
                                                   cuvs::neighbors::quantization::params& params,
                                                   raft::host_matrix_view<const T, int64_t> dataset)
{
  constexpr TempI q_type_min   = static_cast<TempI>(std::numeric_limits<QuantI>::min());
  constexpr TempI q_type_max   = static_cast<TempI>(std::numeric_limits<QuantI>::max());
  constexpr TempI range_q_type = q_type_max - q_type_min + TempI(1);

  size_t n_elements = dataset.extent(0) * dataset.extent(1);

  // conditional: search for quantiles / min / max
  if (!params.is_computed) {
    ASSERT(params.quantile > 0.5 && params.quantile <= 1.0,
           "quantile for scalar quantization needs to be within (.5, 1]");

    double quantile_inv = 1.0 / params.quantile;

    // select subsample
    int seed = 137;
    std::mt19937 rng(seed);
    constexpr size_t num_samples = 10000;
    size_t subset_size           = std::min(num_samples, n_elements);
    std::vector<T> subset(subset_size);
    std::sample(dataset.data_handle(),
                dataset.data_handle() + n_elements,
                std::back_inserter(subset),
                subset_size,
                rng);

    // quantile / sort and pick for now
    thrust::sort(thrust::omp::par, subset.data(), subset.data() + subset_size);

    int pos_max = raft::ceildiv((double)subset_size, quantile_inv) - 1;
    int pos_min = subset_size - pos_max - 1;

    // persist settings in params
    params.min         = double(subset[pos_min]);
    params.max         = double(subset[pos_max]);
    params.scalar      = double(range_q_type) / (params.max - params.min + 1.0);
    params.is_computed = true;
  }

  // allocate target
  auto out = raft::make_host_matrix<QuantI, int64_t>(dataset.extent(0), dataset.extent(1));

#pragma omp parallel for
  for (size_t i = 0; i < n_elements; ++i) {
    TempT tmp_t = ((TempT)dataset.data_handle()[i] - params.min) * params.scalar + TempT(0.5);
    TempI tmp_i =
      raft::max<TempI>(raft::min<TempI>(TempI(tmp_t), q_type_max - q_type_min), TempI(0)) +
      q_type_min;
    out.data_handle()[i] = (QuantI)tmp_i;
  }

  return out;
}

}  // namespace cuvs::neighbors::detail
