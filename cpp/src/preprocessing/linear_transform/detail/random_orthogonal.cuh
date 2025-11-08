/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/linear_transform/random_orthogonal.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/gemm.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/random/rng.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>

namespace cuvs::preprocessing::linear_transform::detail {

template <typename T>
raft::device_matrix<T, int64_t> generate_random_orth(raft::resources const& res,
                                                     const int64_t dim,
                                                     const uint64_t seed)
{
  auto orthogonal_matrix = raft::make_device_matrix<T, int64_t>(res, dim, dim);

  raft::random::RngState rng(seed);
  if constexpr (!std::is_same_v<T, half>) {
    auto rand_matrix = raft::make_device_matrix<T, int64_t>(res, dim, dim);
    raft::random::normal(res,
                         rng,
                         rand_matrix.data_handle(),
                         rand_matrix.size(),
                         static_cast<T>(0),
                         static_cast<T>(1));
    raft::linalg::qrGetQ(res,
                         rand_matrix.data_handle(),
                         orthogonal_matrix.data_handle(),
                         dim,
                         dim,
                         raft::resource::get_cuda_stream(res));
  } else {
    using compute_t            = float;
    auto rand_matrix           = raft::make_device_matrix<float, int64_t>(res, dim, dim);
    auto orthogonal_matrix_f32 = raft::make_device_matrix<compute_t, int64_t>(res, dim, dim);
    raft::random::normal(res,
                         rng,
                         rand_matrix.data_handle(),
                         rand_matrix.size(),
                         static_cast<compute_t>(0),
                         static_cast<compute_t>(1));
    raft::linalg::qrGetQ(res,
                         rand_matrix.data_handle(),
                         orthogonal_matrix_f32.data_handle(),
                         dim,
                         dim,
                         raft::resource::get_cuda_stream(res));

    raft::linalg::map(res,
                      orthogonal_matrix.view(),
                      raft::cast_op<T>{},
                      raft::make_const_mdspan(orthogonal_matrix_f32.view()));
  }

  return orthogonal_matrix;
}

template <typename T>
cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T> train(
  raft::resources const& res,
  const cuvs::preprocessing::linear_transform::random_orthogonal::params params,
  raft::device_matrix_view<const T, int64_t> dataset)
{
  cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T> transformer{
    .orthogonal_matrix = generate_random_orth<T>(res, dataset.extent(1), params.seed)};

  return transformer;
}

template <typename T>
cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T> train(
  raft::resources const& res,
  const cuvs::preprocessing::linear_transform::random_orthogonal::params params,
  raft::host_matrix_view<const T, int64_t> dataset)
{
  cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T> transformer{
    .orthogonal_matrix = generate_random_orth<T>(res, dataset.extent(1), params.seed)};

  return transformer;
}

template <typename T>
void transform(
  raft::resources const& res,
  const cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T>& transformer,
  raft::device_matrix_view<const T, int64_t> dataset,
  raft::device_matrix_view<T, int64_t> out)
{
  RAFT_EXPECTS(dataset.extent(0) == out.extent(0), "Input and output dataset sizes mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == out.extent(1), "Input and output dataset dimensions mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == transformer.orthogonal_matrix.extent(1),
               "Input dataset and transformer dimensions mismatch.");
  RAFT_EXPECTS(transformer.orthogonal_matrix.extent(0) == transformer.orthogonal_matrix.extent(1),
               "Transformer matrix must be square.");

  const auto src_begin = reinterpret_cast<std::uintptr_t>(dataset.data_handle());
  const auto src_end   = src_begin + sizeof(T) * dataset.size();
  const auto dst_begin = reinterpret_cast<std::uintptr_t>(out.data_handle());
  const auto dst_end   = dst_begin + sizeof(T) * out.size();

  // overlapped && src range < dsr range
  const auto overlap = !((src_end < dst_begin) || (dst_end < src_begin));

  const auto dataset_dim = dataset.extent(1);
  const auto orth_view   = raft::make_device_matrix_view<T, int64_t>(
    const_cast<T*>(transformer.orthogonal_matrix.data_handle()), dataset_dim, dataset_dim);
  if (!overlap) {
    // Remove `const`
    auto dataset_view = raft::make_device_matrix_view<T, int64_t>(
      const_cast<T*>(dataset.data_handle()), dataset.extent(0), dataset.extent(1));

    raft::linalg::gemm(res, dataset_view, orth_view, out);
  } else {
    RAFT_EXPECTS(dst_begin <= src_begin,
                 "Must be dst_begin <= src_begin in the current implementation");

    auto mr = raft::resource::get_workspace_resource(res);

    const auto gemm_chunk_size =
      std::min(static_cast<std::uint64_t>(dataset.extent(0)),
               raft::resource::get_workspace_free_bytes(res) / (dataset.extent(1) * sizeof(T)));
    auto neighbor_indices = raft::make_device_mdarray<T, std::int64_t>(
      res, mr, raft::make_extents<std::int64_t>(gemm_chunk_size, dataset.extent(1)));

    raft::spatial::knn::detail::utils::batch_load_iterator<T> dataset_chunk_set(
      dataset.data_handle(),
      dataset.extent(0),
      dataset.extent(1),
      gemm_chunk_size,
      raft::resource::get_cuda_stream(res),
      mr);
    for (auto& batch : dataset_chunk_set) {
      raft::linalg::gemm(res,
                         raft::make_device_matrix_view<T, int64_t>(
                           const_cast<T*>(batch.data()), batch.size(), dataset_dim),
                         orth_view,
                         out);
      raft::copy_async(out.data_handle() + dataset_dim * batch.offset(),
                       batch.data(),
                       batch.size() * dataset_dim,
                       raft::resource::get_cuda_stream(res));
    }
  }
}

template <typename T, typename QuantI = int8_t>
void transform(
  raft::resources const& res,
  const cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T>& transformer,
  raft::host_matrix_view<const T, int64_t> dataset,
  raft::host_matrix_view<QuantI, int64_t> out)
{
  RAFT_EXPECTS(dataset.extent(0) == out.extent(0), "Input and output dataset sizes mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == out.extent(1), "Input and output dataset dimensions mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == transformer.orthogonal_matrix.extent(1),
               "Input dataset and transformer dimensions mismatch.");
  RAFT_EXPECTS(transformer.orthogonal_matrix.extent(0) == transformer.orthogonal_matrix.extent(1),
               "Transformer matrix must be square.");

  auto host_orth = raft::make_host_matrix<T, int64_t>(dataset.extent(1), dataset.extent(1));
  raft::copy(host_orth.data_handle(),
             transformer.orthogonal_matrix.data_handle(),
             host_orth.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

#pragma omp parallel for collapse(2)
  for (int64_t i = 0; i < dataset.extent(0); i++) {
    for (int64_t j = 0; j < dataset.extent(1); j++) {
      auto c = static_cast<T>(0);
      for (int64_t k = 0; k < dataset.extent(1); k++) {
        c += dataset(i, k) * host_orth(k, j);
      }
      out(i, j) = c;
    }
  }
}

template <typename T, typename QuantI = int8_t>
void inverse_transform(
  raft::resources const& res,
  const cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T>& transformer,
  raft::device_matrix_view<const QuantI, int64_t> dataset,
  raft::device_matrix_view<T, int64_t> out)
{
  RAFT_EXPECTS(dataset.extent(0) == out.extent(0), "Input and output dataset sizes mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == out.extent(1), "Input and output dataset dimensions mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == transformer.orthogonal_matrix.extent(1),
               "Input dataset and transformer dimensions mismatch.");
  RAFT_EXPECTS(transformer.orthogonal_matrix.extent(0) == transformer.orthogonal_matrix.extent(1),
               "Transformer matrix must be square.");

  // Remove `const`
  auto dataset_view = raft::make_device_matrix_view<T, int64_t>(
    const_cast<T*>(dataset.data_handle()), dataset.extent(0), dataset.extent(1));

  // Transpose and remove `const`
  const auto dataset_dim = dataset.extent(1);
  const auto orth_T_view = raft::make_device_matrix_view<T, int64_t, raft::layout_f_contiguous>(
    const_cast<T*>(transformer.orthogonal_matrix.data_handle()), dataset_dim, dataset_dim);

  raft::linalg::gemm(res, dataset_view, orth_T_view, out);
}

template <typename T, typename QuantI = int8_t>
void inverse_transform(
  raft::resources const& res,
  const cuvs::preprocessing::linear_transform::random_orthogonal::transformer<T>& transformer,
  raft::host_matrix_view<const QuantI, int64_t> dataset,
  raft::host_matrix_view<T, int64_t> out)
{
  RAFT_EXPECTS(dataset.extent(0) == out.extent(0), "Input and output dataset sizes mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == out.extent(1), "Input and output dataset dimensions mismatch.");
  RAFT_EXPECTS(dataset.extent(1) == transformer.orthogonal_matrix.extent(1),
               "Input dataset and transformer dimensions mismatch.");
  RAFT_EXPECTS(transformer.orthogonal_matrix.extent(0) == transformer.orthogonal_matrix.extent(1),
               "Transformer matrix must be square.");

  auto host_orth = raft::make_host_matrix<T, int64_t>(dataset.extent(1), dataset.extent(1));
  raft::copy(host_orth.data_handle(),
             transformer.orthogonal_matrix.data_handle(),
             host_orth.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

#pragma omp parallel for collapse(2)
  for (int64_t i = 0; i < dataset.extent(0); i++) {
    for (int64_t j = 0; j < dataset.extent(1); j++) {
      auto c = static_cast<T>(0);
      for (int64_t k = 0; k < dataset.extent(1); k++) {
        c += dataset(i, k) * host_orth(j, k);
      }
      out(i, j) = c;
    }
  }
}

}  // namespace cuvs::preprocessing::linear_transform::detail
