/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "../../../neighbors/detail/vpq_dataset.cuh"
#include <cuvs/preprocessing/quantize/product.hpp>
#include <raft/core/operators.hpp>
#include <raft/matrix/init.cuh>

namespace cuvs::preprocessing::quantize::product::detail {
template <typename T>
quantizer<T> train(raft::resources const& res,
                   const cuvs::preprocessing::quantize::product::params params,
                   raft::device_matrix_view<const T, int64_t> dataset)
{
  auto n_rows = dataset.extent(0);
  auto dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::product::train(%zu, %u)", size_t(n_rows), dim);

  auto ps = cuvs::neighbors::detail::fill_missing_params_heuristics(params, dataset);
  std::optional<raft::device_matrix_view<const T, uint32_t, raft::row_major>>
    vq_code_book_view_opt                                                           = std::nullopt;
  std::optional<raft::device_matrix<T, uint32_t, raft::row_major>> vq_code_book_opt = std::nullopt;
  if (params.vq_n_centers == 1) {
    vq_code_book_opt = raft::make_device_matrix<T, uint32_t, raft::row_major>(res, 0, 0);
  } else {
    vq_code_book_opt      = cuvs::neighbors::detail::train_vq<T>(res, ps, dataset);
    vq_code_book_view_opt = raft::make_const_mdspan(vq_code_book_opt.value().view());
  }

  auto pq_code_book = cuvs::neighbors::detail::train_pq<T>(res, ps, dataset, vq_code_book_view_opt);
  auto empty_codes  = raft::make_device_matrix<uint8_t, int64_t, raft::row_major>(res, 0, 0);
  return {ps,
          cuvs::neighbors::vpq_dataset<T, int64_t>{
            std::move(vq_code_book_opt.value()), std::move(pq_code_book), std::move(empty_codes)}};
}

template <typename T, typename QuantI = uint8_t>
void transform(raft::resources const& res,
               const quantizer<T>& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  RAFT_EXPECTS(out.extent(0) == dataset.extent(0),
               "Output matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(out.extent(1) == cuvs::preprocessing::quantize::product::get_quantized_dim<uint32_t>(
                                  quantizer.params_quantizer),
               "Output matrix doesn't have the correct number of columns");
  // Encode dataset
  std::optional<raft::device_matrix_view<const T, uint32_t, raft::row_major>> vq_centers =
    std::nullopt;
  if (quantizer.params_quantizer.vq_n_centers != 1) {
    vq_centers = raft::make_const_mdspan(quantizer.vpq_codebooks.vq_code_book.view());
  }

  cuvs::neighbors::detail::process_and_fill_codes<T, int64_t>(
    res,
    quantizer.params_quantizer,
    dataset,
    vq_centers,
    raft::make_const_mdspan(quantizer.vpq_codebooks.pq_code_book.view()),
    out);
}
}  // namespace cuvs::preprocessing::quantize::product::detail
