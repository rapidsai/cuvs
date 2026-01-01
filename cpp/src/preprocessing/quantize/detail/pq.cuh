/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../core/nvtx.hpp"
#include "../../../neighbors/detail/vpq_dataset.cuh"
#include "../../../neighbors/ivf_pq/ivf_pq_codepacking.cuh"  // pq_bits-bitfield

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/preprocessing/quantize/pq.hpp>
#include <raft/core/operators.hpp>
#include <raft/matrix/init.cuh>

#include "../../../cluster/kmeans_balanced.cuh"

namespace cuvs::preprocessing::quantize::pq::detail {

template <typename MathT, typename DatasetT>
auto train_pq_subspaces(
  const raft::resources& res,
  const cuvs::preprocessing::quantize::pq::params& params,
  const DatasetT& dataset,
  std::optional<const raft::device_matrix_view<const MathT, uint32_t, raft::row_major>> vq_centers)
  -> raft::device_matrix<MathT, uint32_t, raft::row_major>
{
  using ix_t              = int64_t;
  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t pq_len       = raft::div_rounding_up_safe(dim, pq_dim);
  RAFT_EXPECTS((dim % pq_dim) == 0, "Dimension must be divisible by pq_dim");
  ix_t n_rows_train = n_rows * params.pq_kmeans_trainset_fraction;
  n_rows_train      = std::min(n_rows_train, params.max_train_points_per_pq_code * pq_n_centers);
  RAFT_EXPECTS(
    n_rows_train >= pq_n_centers,
    "The number of training samples must be equal to or greater than the number of PQ centers");

  std::optional<raft::device_matrix<MathT, ix_t, raft::row_major>> pq_trainset = std::nullopt;

  // Subtract VQ centers
  if (vq_centers) {
    pq_trainset = std::make_optional(cuvs::util::subsample(res, dataset, n_rows_train));
    auto vq_labels =
      cuvs::neighbors::detail::predict_vq<uint32_t>(res, pq_trainset.value(), vq_centers.value());
    using index_type = typename DatasetT::index_type;
    raft::linalg::map_offset(
      res,
      pq_trainset.value().view(),
      [labels = vq_labels.view(), centers = vq_centers.value(), dim] __device__(index_type off,
                                                                                MathT x) {
        index_type i = off / dim;
        index_type j = off % dim;
        return x - centers(labels(i), j);
      },
      raft::make_const_mdspan(pq_trainset.value().view()));
  }

  // Train PQ centers for each subspace
  auto sub_dataset = raft::make_device_matrix<MathT, ix_t>(res, n_rows_train, pq_len);

  auto pq_centers =
    raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, pq_dim * pq_n_centers, pq_len);
  auto trainset_ptr = pq_trainset ? pq_trainset.value().data_handle() : dataset.data_handle();
  std::optional<rmm::device_uvector<uint32_t>> sub_labels_opt                       = std::nullopt;
  std::optional<rmm::device_uvector<uint32_t>> pq_cluster_sizes_opt                 = std::nullopt;
  std::optional<raft::device_vector_view<uint32_t, ix_t>> sub_labels_view_opt       = std::nullopt;
  std::optional<raft::device_vector_view<uint32_t, ix_t>> pq_cluster_sizes_view_opt = std::nullopt;
  auto device_memory = raft::resource::get_workspace_resource(res);
  if (params.pq_kmeans_type == cuvs::cluster::kmeans::kmeans_type::KMeansBalanced) {
    auto stream = raft::resource::get_cuda_stream(res);
    sub_labels_opt =
      std::make_optional(rmm::device_uvector<uint32_t>(n_rows_train, stream, device_memory));
    pq_cluster_sizes_opt =
      std::make_optional(rmm::device_uvector<uint32_t>(pq_n_centers, stream, device_memory));
    sub_labels_view_opt = std::make_optional(
      raft::make_device_vector_view<uint32_t, ix_t>(sub_labels_opt.value().data(), n_rows_train));
    pq_cluster_sizes_view_opt = std::make_optional(raft::make_device_vector_view<uint32_t, ix_t>(
      pq_cluster_sizes_opt.value().data(), pq_n_centers));
  }

  for (ix_t m = 0; m < pq_dim; m++) {
    RAFT_CUDA_TRY(cudaMemcpy2DAsync(sub_dataset.data_handle(),
                                    sizeof(MathT) * pq_len,
                                    trainset_ptr + m * pq_len,
                                    sizeof(MathT) * dim,
                                    sizeof(MathT) * pq_len,
                                    n_rows_train,
                                    cudaMemcpyDefault,
                                    raft::resource::get_cuda_stream(res)));
    auto pq_centers_subspace_view = raft::make_device_matrix_view<MathT, uint32_t, raft::row_major>(
      pq_centers.data_handle() + m * pq_n_centers * pq_len, pq_n_centers, pq_len);
    cuvs::neighbors::detail::train_pq_centers<MathT, ix_t>(
      res,
      params,
      raft::make_const_mdspan(sub_dataset.view()),
      pq_centers_subspace_view,
      sub_labels_view_opt,
      pq_cluster_sizes_view_opt);
  }
  return pq_centers;
}

template <typename DataT, typename MathT, typename AccessorType>
quantizer<MathT> train(
  raft::resources const& res,
  const cuvs::preprocessing::quantize::pq::params params,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, AccessorType> dataset)
{
  auto n_rows = dataset.extent(0);
  auto dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::train(%zu, %u, %u, %u)",
    size_t(n_rows),
    dim,
    params.pq_bits,
    params.pq_dim);

  cuvs::preprocessing::quantize::pq::params ps(
    cuvs::neighbors::detail::fill_missing_params_heuristics(
      static_cast<const cuvs::neighbors::vpq_params&>(params), dataset),
    params.use_vq,
    params.use_subspaces);
  std::optional<raft::device_matrix_view<const MathT, uint32_t, raft::row_major>>
    vq_code_book_view_opt = std::nullopt;
  std::optional<raft::device_matrix<MathT, uint32_t, raft::row_major>> vq_code_book_opt =
    std::nullopt;
  if (ps.use_vq) {
    vq_code_book_opt      = cuvs::neighbors::detail::train_vq<MathT>(res, ps, dataset);
    vq_code_book_view_opt = raft::make_const_mdspan(vq_code_book_opt.value().view());
  } else {
    vq_code_book_opt = raft::make_device_matrix<MathT, uint32_t, raft::row_major>(res, 0, 0);
  }
  auto empty_codes = raft::make_device_matrix<uint8_t, int64_t, raft::row_major>(res, 0, 0);
  if (ps.use_subspaces) {
    auto pq_code_book = train_pq_subspaces<MathT>(res, ps, dataset, vq_code_book_view_opt);
    return {
      ps,
      cuvs::neighbors::vpq_dataset<MathT, int64_t>{
        std::move(vq_code_book_opt.value()), std::move(pq_code_book), std::move(empty_codes)}};
  } else {
    auto pq_code_book =
      cuvs::neighbors::detail::train_pq<MathT>(res, ps, dataset, vq_code_book_view_opt);
    return {
      ps,
      cuvs::neighbors::vpq_dataset<MathT, int64_t>{
        std::move(vq_code_book_opt.value()), std::move(pq_code_book), std::move(empty_codes)}};
  }
}

template <typename T, typename QuantI, typename AccessorType>
void transform(
  raft::resources const& res,
  const quantizer<T>& quantizer,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, AccessorType> dataset,
  raft::device_matrix_view<QuantI, int64_t> out)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::transform(%zu, %zu, %zu)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    size_t(out.extent(1)));
  RAFT_EXPECTS(out.extent(0) == dataset.extent(0),
               "Output matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(out.extent(1) == get_quantized_dim(quantizer.params_quantizer),
               "Output matrix doesn't have the correct number of columns");
  // Encode dataset
  std::optional<raft::device_matrix_view<const T, uint32_t, raft::row_major>> vq_centers =
    std::nullopt;
  if (quantizer.params_quantizer.use_vq) {
    vq_centers = raft::make_const_mdspan(quantizer.vpq_codebooks.vq_code_book.view());
  }

  if (quantizer.params_quantizer.use_subspaces) {
    cuvs::neighbors::detail::process_and_fill_codes_subspaces<T, int64_t>(
      res,
      quantizer.params_quantizer,
      dataset,
      vq_centers,
      raft::make_const_mdspan(quantizer.vpq_codebooks.pq_code_book.view()),
      out);
  } else {
    cuvs::neighbors::detail::process_and_fill_codes<T, int64_t>(
      res,
      quantizer.params_quantizer,
      dataset,
      vq_centers,
      raft::make_const_mdspan(quantizer.vpq_codebooks.pq_code_book.view()),
      out);
  }
}

template <uint32_t BlockSize,
          uint32_t PqBits,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL reconstruct_vectors_kernel(
  raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> codes,
  raft::device_matrix_view<DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  std::optional<raft::device_matrix_view<const DataT, IdxT, raft::row_major>> vq_centers,
  bool use_subspaces)
{
  const uint32_t kSubWarpSize  = raft::WarpSize;
  constexpr uint32_t n_centers = 1 << PqBits;
  using subwarp_align          = raft::Pow2<kSubWarpSize>;
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= dataset.extent(0)) { return; }
  uint32_t lane_id      = subwarp_align::mod(raft::laneId());
  const uint32_t pq_len = raft::div_rounding_up_unsafe(dataset.extent(1), pq_centers.extent(1));

  const uint8_t* out_codes_ptr = &codes(row_ix, 0);
  LabelT vq_label              = 0;
  if (vq_centers) {
    vq_label      = *reinterpret_cast<const LabelT*>(out_codes_ptr);
    out_codes_ptr = (&codes(row_ix, 0)) + sizeof(LabelT);
  }
  cuvs::neighbors::ivf_pq::detail::bitfield_view_t<PqBits, const uint8_t> code_view{out_codes_ptr};
  for (uint32_t j = lane_id; j < pq_len; j += kSubWarpSize) {
    const uint8_t code = code_view[j];
    for (uint32_t k = 0; k < pq_centers.extent(1); k++) {
      const auto col                    = j * pq_centers.extent(1) + k;
      const uint32_t pq_subspace_offset = use_subspaces ? n_centers * j : 0;
      if (vq_centers) {
        dataset(row_ix, col) =
          pq_centers(pq_subspace_offset + code, k) + vq_centers.value()(vq_label, col);
      } else {
        dataset(row_ix, col) = pq_centers(pq_subspace_offset + code, k);
      }
    }
  }
}

template <typename DataT, typename MathT, typename IdxT, typename LabelT>
auto reconstruct_vectors(
  const raft::resources& res,
  const params& params,
  raft::device_matrix_view<const uint8_t, IdxT, raft::row_major> codes,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  std::optional<raft::device_matrix_view<const DataT, uint32_t, raft::row_major>> vq_centers,
  raft::device_matrix_view<DataT, IdxT, raft::row_major> out_vectors,
  bool use_subspaces)
{
  using data_t = DataT;
  using ix_t   = IdxT;

  const ix_t n_rows       = out_vectors.extent(0);
  const ix_t dim          = out_vectors.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;

  auto stream = raft::resource::get_cuda_stream(res);

  constexpr ix_t kBlockSize  = 256;
  const ix_t threads_per_vec = raft::WarpSize;
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4: return reconstruct_vectors_kernel<kBlockSize, 4, data_t, MathT, IdxT, LabelT>;
      case 5: return reconstruct_vectors_kernel<kBlockSize, 5, data_t, MathT, IdxT, LabelT>;
      case 6: return reconstruct_vectors_kernel<kBlockSize, 6, data_t, MathT, IdxT, LabelT>;
      case 7: return reconstruct_vectors_kernel<kBlockSize, 7, data_t, MathT, IdxT, LabelT>;
      case 8: return reconstruct_vectors_kernel<kBlockSize, 8, data_t, MathT, IdxT, LabelT>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(pq_bits);
  dim3 blocks(raft::div_rounding_up_safe<ix_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);
  kernel<<<blocks, threads, 0, stream>>>(codes, out_vectors, pq_centers, vq_centers, use_subspaces);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  return codes;
}

template <typename T, typename QuantI = uint8_t>
void inverse_transform(raft::resources const& res,
                       const quantizer<T>& quant,
                       raft::device_matrix_view<const QuantI, int64_t> codes,
                       raft::device_matrix_view<T, int64_t> out)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::inverse_transform(%zu, %zu, %zu)",
    size_t(codes.extent(0)),
    size_t(codes.extent(1)),
    size_t(out.extent(1)));
  using label_t = uint32_t;
  using idx_t   = int64_t;
  RAFT_EXPECTS(out.extent(0) == codes.extent(0),
               "Output matrix must have the same number of rows as the input codes");
  RAFT_EXPECTS(codes.extent(1) == get_quantized_dim(quant.params_quantizer),
               "Codes matrix doesn't have the correct number of columns");

  std::optional<raft::device_matrix_view<const T, uint32_t, raft::row_major>> vq_centers_opt =
    std::nullopt;
  if (quant.params_quantizer.use_vq) {
    vq_centers_opt = raft::make_const_mdspan(quant.vpq_codebooks.vq_code_book.view());
  }

  reconstruct_vectors<T, T, idx_t, label_t>(
    res,
    quant.params_quantizer,
    codes,
    raft::make_const_mdspan(quant.vpq_codebooks.pq_code_book.view()),
    vq_centers_opt,
    out,
    quant.params_quantizer.use_subspaces);
}
}  // namespace cuvs::preprocessing::quantize::pq::detail
