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

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>

namespace cuvs::neighbors::quantization {
struct params {
  float quantile = 0.99;

  bool is_computed = false;
  double min;
  double max;
  double scalar;
};

raft::device_matrix<int8_t, int64_t> scalar_quantize(
  raft::resources const& res,
  cuvs::neighbors::quantization::params& params,
  raft::device_matrix_view<const double, int64_t> dataset);

raft::device_matrix<int8_t, int64_t> scalar_quantize(
  raft::resources const& res,
  cuvs::neighbors::quantization::params& params,
  raft::device_matrix_view<const float, int64_t> dataset);

raft::device_matrix<int8_t, int64_t> scalar_quantize(
  raft::resources const& res,
  cuvs::neighbors::quantization::params& params,
  raft::device_matrix_view<const half, int64_t> dataset);

raft::host_matrix<int8_t, int64_t> scalar_quantize(
  raft::resources const& res,
  cuvs::neighbors::quantization::params& params,
  raft::host_matrix_view<const double, int64_t> dataset);

raft::host_matrix<int8_t, int64_t> scalar_quantize(
  raft::resources const& res,
  cuvs::neighbors::quantization::params& params,
  raft::host_matrix_view<const float, int64_t> dataset);

raft::host_matrix<int8_t, int64_t> scalar_quantize(
  raft::resources const& res,
  cuvs::neighbors::quantization::params& params,
  raft::host_matrix_view<const half, int64_t> dataset);

}  // namespace cuvs::neighbors::quantization
