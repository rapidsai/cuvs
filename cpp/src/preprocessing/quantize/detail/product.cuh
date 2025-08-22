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

#pragma once

#include "../../../core/nvtx.hpp"
#include <cuvs/preprocessing/quantize/product.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/init.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>

namespace cuvs::preprocessing::quantize::product::detail {

template <typename T>
quantizer train(raft::resources const& res,
                const cuvs::preprocessing::quantize::product::params params,
                raft::device_matrix_view<const T, int64_t> dataset)
{
  auto n_rows = dataset.extent(0);
  auto dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::product::train(%zu, %u)", size_t(n_rows), dim);

  auto pq_params                         = cuvs::neighbors::ivf_pq::index_params();
  pq_params.n_lists                      = 1;
  pq_params.pq_bits                      = params.pq_bits;
  pq_params.pq_dim                       = params.pq_dim;
  pq_params.codebook_kind                = params.codebook_kind;
  pq_params.force_random_rotation        = params.force_random_rotation;
  pq_params.add_data_on_build            = false;
  pq_params.max_train_points_per_pq_code = params.max_train_points_per_pq_code;

  auto pq_index = cuvs::neighbors::ivf_pq::build(res, pq_params, dataset);
  return cuvs::preprocessing::quantize::product::quantizer{std::move(pq_index)};
}

quantizer train(
  raft::resources const& res,
  const cuvs::preprocessing::quantize::product::params params,
  const uint32_t dim,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
  std::optional<raft::device_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
  std::optional<raft::device_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::product::train()");
  auto pq_params                         = cuvs::neighbors::ivf_pq::index_params();
  pq_params.n_lists                      = centers.extent(0);
  pq_params.pq_bits                      = params.pq_bits;
  pq_params.pq_dim                       = params.pq_dim;
  pq_params.codebook_kind                = params.codebook_kind;
  pq_params.force_random_rotation        = params.force_random_rotation;
  pq_params.max_train_points_per_pq_code = params.max_train_points_per_pq_code;

  auto pq_index = cuvs::neighbors::ivf_pq::build(
    res, pq_params, dim, pq_centers, centers, centers_rot, rotation_matrix);
  return cuvs::preprocessing::quantize::product::quantizer{std::move(pq_index)};
}

template <typename T, typename QuantI = uint8_t>
void transform(raft::resources const& res,
               const quantizer& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  // std::optional<raft::device_vector<int64_t, int64_t>> indices_opt = std::nullopt;
  std::optional<raft::device_vector_view<const int64_t, int64_t>> indices_view_opt = std::nullopt;
  // auto current_size = quantizer.pq_index.size();
  // if (current_size != 0) {
  //   indices_opt = raft::make_device_vector<int64_t>(res, dataset.extent(0));
  //   raft::linalg::range(indices_opt.value().data_handle(), current_size,
  //                       current_size + dataset.extent(0), raft::resource::get_cuda_stream(res));
  //   indices_view_opt = raft::make_const_mdspan(indices_opt.value().view());
  // }

  // TODO: Call detail::extend to avoid clone()
  auto extended_index =
    cuvs::neighbors::ivf_pq::extend(res, dataset, indices_view_opt, quantizer.pq_index);

  auto n_lists        = extended_index.n_lists();
  auto lists          = extended_index.lists();
  auto indices_uint32 = rmm::device_uvector<uint32_t>(0, raft::resource::get_cuda_stream(res));
  for (size_t i = 0; i < n_lists; i++) {
    /*auto current_list_size = lists[i]->size.load();
    if (current_list_size > indices_uint32.size()) {
      indices_uint32.resize(current_list_size, raft::resource::get_cuda_stream(res));
    }
    auto indices_uint32_view = raft::make_device_vector_view<uint32_t>(indices_uint32.data(),
    current_list_size); auto indices_list_view = raft::make_device_vector_view<const
    int64_t>(lists[i]->indices.data_handle(), current_list_size); raft::linalg::map(res,
    indices_uint32_view, raft::cast_op<uint32_t>{}, indices_list_view);

    cuvs::neighbors::ivf_pq::helpers::codepacker::unpack_list_data(
      res, extended_index, raft::make_const_mdspan(indices_uint32_view), out, i);*/

    cuvs::neighbors::ivf_pq::helpers::codepacker::unpack_list_data(res, extended_index, out, i, 0);
  }
  // TODO: Resize the extended index lists to 0
}

/*
template <typename T>
quantizer train(
  raft::resources const& res,
  const params params,
  raft::host_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(params.quantile > 0.0 && params.quantile <= 1.0,
               "quantile for scalar quantization needs to be within (0, 1] but is %f",
               params.quantile);

  auto [min, max] = detail::quantile_min_max(res, dataset, params.quantile);

  RAFT_LOG_DEBUG("quantizer train min=%lf max=%lf.", double(min), double(max));

  return quantizer{min, max};
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               const quantizer& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);

  raft::linalg::map(res, out, quantize_op<T, QuantI>(quantizer.min_, quantizer.max_), dataset);
}

template <typename T, typename QuantI = int8_t>
void transform(raft::resources const& res,
               const quantizer& quantizer,
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
*/
}  // namespace cuvs::preprocessing::quantize::product::detail
