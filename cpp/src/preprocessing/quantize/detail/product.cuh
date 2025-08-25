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
#include <raft/matrix/scatter.cuh>
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
  pq_params.n_lists                      = params.n_lists;
  pq_params.pq_bits                      = params.pq_bits;
  pq_params.pq_dim                       = params.pq_dim;
  pq_params.codebook_kind                = params.codebook_kind;
  pq_params.force_random_rotation        = params.force_random_rotation;
  pq_params.add_data_on_build            = false;
  pq_params.max_train_points_per_pq_code = params.max_train_points_per_pq_code;

  auto pq_index = cuvs::neighbors::ivf_pq::build(res, pq_params, dataset);
  return cuvs::preprocessing::quantize::product::quantizer{std::move(pq_index)};
}

template <typename T, typename QuantI = uint8_t>
void transform(raft::resources const& res,
               const quantizer& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::product::transform()");
  std::optional<raft::device_vector_view<const int64_t, int64_t>> indices_view_opt = std::nullopt;

  auto extended_index =
    cuvs::neighbors::ivf_pq::extend(res, dataset, indices_view_opt, quantizer.pq_index);

  auto n_lists = extended_index.n_lists();
  auto lists   = extended_index.lists();
  if (n_lists == 1) {
    cuvs::neighbors::ivf_pq::helpers::codepacker::unpack_list_data(res, extended_index, out, 0, 0);
  } else {
    auto indices = raft::make_device_vector<int64_t, int64_t>(res, dataset.extent(0));
    auto offset  = 0;
    for (size_t i = 0; i < n_lists; i++) {
      auto current_list_size = lists[i]->size.load();

      auto current_indices_view =
        raft::make_device_vector_view<int64_t>(indices.data_handle() + offset, current_list_size);
      auto current_out_view = raft::make_device_matrix_view<QuantI, int64_t>(
        out.data_handle() + offset * out.extent(1), current_list_size, out.extent(1));

      auto indices_list_view = raft::make_device_vector_view<const int64_t>(
        lists[i]->indices.data_handle(), current_list_size);
      raft::linalg::map(res, current_indices_view, raft::identity_op{}, indices_list_view);

      cuvs::neighbors::ivf_pq::helpers::codepacker::unpack_list_data(
        res, extended_index, current_out_view, i, 0);

      offset += current_list_size;
    }

    raft::matrix::scatter(res, out, raft::make_const_mdspan(indices.view()));
  }
}

}  // namespace cuvs::preprocessing::quantize::product::detail
