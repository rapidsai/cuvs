/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../core/nvtx.hpp"
#include "../../../neighbors/detail/vpq_dataset.cuh"
#include "../../../neighbors/ivf_pq/ivf_pq_codepacking.cuh"  // pq_bits-bitfield
#include <cuvs/preprocessing/quantize/pq.hpp>
#include <raft/core/operators.hpp>
#include <raft/matrix/init.cuh>

namespace cuvs::preprocessing::quantize::pq::detail {
template <typename T>
quantizer<T> train(raft::resources const& res,
                   const cuvs::preprocessing::quantize::pq::params params,
                   raft::device_matrix_view<const T, int64_t> dataset)
{
  auto n_rows = dataset.extent(0);
  auto dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::train(%zu, %u)", size_t(n_rows), dim);

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
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::pq::transform(%zu, %zu, %zu)",
    size_t(dataset.extent(0)),
    size_t(dataset.extent(1)),
    size_t(out.extent(1)));
  RAFT_EXPECTS(out.extent(0) == dataset.extent(0),
               "Output matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(out.extent(1) == cuvs::preprocessing::quantize::pq::get_quantized_dim<uint32_t>(
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
  const uint32_t pq_dim,
  std::optional<raft::device_matrix_view<const DataT, IdxT, raft::row_major>> vq_centers)
{
  const uint32_t kSubWarpSize = raft::WarpSize;
  using subwarp_align         = raft::Pow2<kSubWarpSize>;
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= dataset.extent(0)) { return; }
  uint32_t lane_id = subwarp_align::mod(raft::laneId());

  const uint8_t* out_codes_ptr = &codes(row_ix, 0);
  LabelT vq_label              = *reinterpret_cast<const LabelT*>(out_codes_ptr);
  out_codes_ptr                = (&codes(row_ix, 0)) + sizeof(LabelT);
  cuvs::neighbors::ivf_pq::detail::bitfield_view_t<PqBits, const uint8_t> code_view{out_codes_ptr};
  for (uint32_t j = lane_id; j < pq_dim; j += kSubWarpSize) {
    uint8_t code = code_view[j];
    for (uint32_t k = 0; k < pq_centers.extent(1); k++) {
      const auto col = j * pq_centers.extent(1) + k;
      if (vq_centers) {
        dataset(row_ix, col) = pq_centers(code, k) + vq_centers.value()(vq_label, col);
      } else {
        dataset(row_ix, col) = pq_centers(code, k);
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
  raft::device_matrix_view<DataT, IdxT, raft::row_major> out_vectors)
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
  kernel<<<blocks, threads, 0, stream>>>(codes, out_vectors, pq_centers, pq_dim, vq_centers);
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
  RAFT_EXPECTS(codes.extent(1) == cuvs::preprocessing::quantize::pq::get_quantized_dim<label_t>(
                                    quant.params_quantizer),
               "Codes matrix doesn't have the correct number of columns");

  std::optional<raft::device_matrix_view<const T, uint32_t, raft::row_major>> vq_centers_opt =
    std::nullopt;
  if (quant.params_quantizer.vq_n_centers != 1) {
    vq_centers_opt = raft::make_const_mdspan(quant.vpq_codebooks.vq_code_book.view());
  }

  reconstruct_vectors<T, T, idx_t, label_t>(
    res,
    quant.params_quantizer,
    codes,
    raft::make_const_mdspan(quant.vpq_codebooks.pq_code_book.view()),
    vq_centers_opt,
    out);
}
}  // namespace cuvs::preprocessing::quantize::pq::detail
