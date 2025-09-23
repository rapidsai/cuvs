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
#include <raft/linalg/init.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/matrix/scatter.cuh>
#include <raft/random/rng.cuh>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>

namespace cuvs::neighbors::detail {

template <uint32_t BlockSize,
          uint32_t PqBits,
          typename DataT,
          typename MathT,
          typename IdxT,
          typename LabelT>
__launch_bounds__(BlockSize) RAFT_KERNEL process_and_fill_codes_no_id_kernel(
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> out_codes,
  raft::device_matrix_view<const DataT, IdxT, raft::row_major> dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_vector_view<const LabelT, IdxT, raft::row_major> vq_labels,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers)
{
  constexpr uint32_t kSubWarpSize = std::min<uint32_t>(raft::WarpSize, 1u << PqBits);
  using subwarp_align             = raft::Pow2<kSubWarpSize>;
  const IdxT row_ix = subwarp_align::div(IdxT{threadIdx.x} + IdxT{BlockSize} * IdxT{blockIdx.x});
  if (row_ix >= out_codes.extent(0)) { return; }

  const uint32_t pq_dim = raft::div_rounding_up_unsafe(vq_centers.extent(1), pq_centers.extent(1));

  const uint32_t lane_id = raft::Pow2<kSubWarpSize>::mod(threadIdx.x);
  const LabelT vq_label  = vq_labels(row_ix);

  // write codes
  auto* out_codes_ptr = &out_codes(row_ix, 0);
  cuvs::neighbors::ivf_pq::detail::bitfield_view_t<PqBits> code_view{out_codes_ptr};
  for (uint32_t j = 0; j < pq_dim; j++) {
    // find PQ label
    uint8_t code = compute_code<kSubWarpSize>(dataset, vq_centers, pq_centers, row_ix, j, vq_label);
    // TODO: this writes in global memory one byte per warp, which is very slow.
    //  It's better to keep the codes in the shared memory or registers and dump them at once.
    if (lane_id == 0) { code_view[j] = code; }
  }
}

template <typename MathT, typename IdxT, typename DatasetT>
void process_and_fill_codes_no_id(
  const raft::resources& res,
  const vpq_params& params,
  const DatasetT& dataset,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> vq_centers,
  raft::device_matrix_view<const MathT, uint32_t, raft::row_major> pq_centers,
  raft::device_matrix_view<uint8_t, IdxT, raft::row_major> codes)
{
  using data_t     = typename DatasetT::value_type;
  using cdataset_t = vpq_dataset<MathT, IdxT>;
  using label_t    = uint32_t;

  const ix_t n_rows       = dataset.extent(0);
  const ix_t dim          = dataset.extent(1);
  const ix_t pq_dim       = params.pq_dim;
  const ix_t pq_bits      = params.pq_bits;
  const ix_t pq_n_centers = ix_t{1} << pq_bits;
  const ix_t codes_rowlen = raft::div_rounding_up_safe<ix_t>(pq_dim * pq_bits, 8);
  RAFT_EXPECTS(codes.extent(0) == n_rows,
               "Codes matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(codes.extent(1) == codes_rowlen,
               "Codes matrix must have the same number of columns as the input dataset");

  auto stream = raft::resource::get_cuda_stream(res);

  // TODO: with scaling workspace we could choose the batch size dynamically
  constexpr ix_t kReasonableMaxBatchSize = 65536;
  constexpr ix_t kBlockSize              = 256;
  const ix_t threads_per_vec             = std::min<ix_t>(raft::WarpSize, pq_n_centers);
  dim3 threads(kBlockSize, 1, 1);
  ix_t max_batch_size = std::min<ix_t>(n_rows, kReasonableMaxBatchSize);
  auto kernel         = [](uint32_t pq_bits) {
    switch (pq_bits) {
      case 4:
        return process_and_fill_codes_no_id_kernel<kBlockSize, 4, data_t, MathT, IdxT, label_t>;
      case 5:
        return process_and_fill_codes_no_id_kernel<kBlockSize, 5, data_t, MathT, IdxT, label_t>;
      case 6:
        return process_and_fill_codes_no_id_kernel<kBlockSize, 6, data_t, MathT, IdxT, label_t>;
      case 7:
        return process_and_fill_codes_no_id_kernel<kBlockSize, 7, data_t, MathT, IdxT, label_t>;
      case 8:
        return process_and_fill_codes_no_id_kernel<kBlockSize, 8, data_t, MathT, IdxT, label_t>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }(pq_bits);
  for (const auto& batch : cuvs::spatial::knn::detail::utils::batch_load_iterator(
         dataset.data_handle(),
         n_rows,
         dim,
         max_batch_size,
         stream,
         rmm::mr::get_current_device_resource())) {
    auto batch_view = raft::make_device_matrix_view(batch.data(), ix_t(batch.size()), dim);

    auto labels = raft::make_device_vector<label_t, ix_t>(res, batch.size());
    raft::matrix::fill(res, labels.view(), uint32_t{0});
    dim3 blocks(raft::div_rounding_up_safe<ix_t>(n_rows, kBlockSize / threads_per_vec), 1, 1);
    kernel<<<blocks, threads, 0, stream>>>(
      raft::make_device_matrix_view<uint8_t, IdxT>(
        codes.data_handle() + batch.offset() * codes_rowlen, batch.size(), codes_rowlen),
      batch_view,
      vq_centers,
      raft::make_const_mdspan(labels.view()),
      pq_centers);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}
}  // namespace cuvs::neighbors::detail

namespace cuvs::preprocessing::quantize::product::detail {
template <typename T>
quantizer<T> train(raft::resources const& res,
                   const cuvs::preprocessing::quantize::product::params params,
                   raft::device_matrix_view<const T, int64_t> dataset)
{
  RAFT_EXPECTS(params.vq_n_centers == 1, "Only vq_n_centers = 1 is supported");

  auto n_rows = dataset.extent(0);
  auto dim    = dataset.extent(1);
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "preprocessing::quantize::product::train(%zu, %u)", size_t(n_rows), dim);

  auto ps = cuvs::neighbors::detail::fill_missing_params_heuristics(params, dataset);
  auto vq_code_book =
    raft::make_device_matrix<T, uint32_t, raft::row_major>(res, ps.vq_n_centers, dim);
  raft::matrix::fill(res, vq_code_book.view(), 0.0f);
  auto pq_code_book = cuvs::neighbors::detail::train_pq<T>(res, ps, dataset, std::nullopt);
  auto empty_codes  = raft::make_device_matrix<uint8_t, int64_t, raft::row_major>(res, 0, 0);
  return {ps,
          cuvs::neighbors::vpq_dataset<T, int64_t>{
            std::move(vq_code_book), std::move(pq_code_book), std::move(empty_codes)}};
}

template <typename T, typename QuantI = uint8_t>
void transform(raft::resources const& res,
               const quantizer<T>& quantizer,
               raft::device_matrix_view<const T, int64_t> dataset,
               raft::device_matrix_view<QuantI, int64_t> out)
{
  RAFT_EXPECTS(out.extent(0) == dataset.extent(0),
               "Output matrix must have the same number of rows as the input dataset");
  RAFT_EXPECTS(
    out.extent(1) == raft::div_rounding_up_safe<int64_t>(
                       quantizer.vpq_dataset.pq_bits() * quantizer.vpq_dataset.pq_dim(), 8),
    "Output matrix must have (pq_dim * pq_bits / 8) columns");
  // Encode dataset
  cuvs::neighbors::detail::process_and_fill_codes_no_id<T, int64_t>(
    res,
    quantizer.params,
    dataset,
    raft::make_const_mdspan(quantizer.vpq_dataset.vq_code_book.view()),
    raft::make_const_mdspan(quantizer.vpq_dataset.pq_code_book.view()),
    out);
}
}  // namespace cuvs::preprocessing::quantize::product::detail
