/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/pq.cuh"

#include <cuvs/preprocessing/quantize/pq.hpp>

#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <variant>

namespace cuvs::preprocessing::quantize::pq {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                              \
  auto build(raft::resources const& res,                                               \
             const params params,                                                      \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>       \
  {                                                                                    \
    return detail::build<T, T>(res, params, dataset);                                  \
  }                                                                                    \
  auto build(raft::resources const& res,                                               \
             const params params,                                                      \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>         \
  {                                                                                    \
    return detail::build<T, T>(res, params, dataset);                                  \
  }                                                                                    \
  void transform(raft::resources const& res,                                           \
                 const quantizer<T>& quantizer,                                        \
                 raft::device_matrix_view<const T, int64_t> dataset,                   \
                 raft::device_matrix_view<QuantI, int64_t> codes_out,                  \
                 std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels) \
  {                                                                                    \
    detail::transform(res, quantizer, dataset, codes_out, vq_labels);                  \
  }                                                                                    \
  void transform(raft::resources const& res,                                           \
                 const quantizer<T>& quantizer,                                        \
                 raft::host_matrix_view<const T, int64_t> dataset,                     \
                 raft::device_matrix_view<QuantI, int64_t> codes_out,                  \
                 std::optional<raft::device_vector_view<uint32_t, int64_t>> vq_labels) \
  {                                                                                    \
    detail::transform(res, quantizer, dataset, codes_out, vq_labels);                  \
  }                                                                                    \
  void inverse_transform(                                                              \
    raft::resources const& res,                                                        \
    const quantizer<T>& quantizer,                                                     \
    raft::device_matrix_view<const QuantI, int64_t> pq_codes,                          \
    raft::device_matrix_view<T, int64_t> out,                                          \
    std::optional<raft::device_vector_view<const uint32_t, int64_t>> vq_labels)        \
  {                                                                                    \
    detail::inverse_transform(res, quantizer, pq_codes, out, vq_labels);               \
  }

CUVS_INST_QUANTIZATION(float, uint8_t);

#undef CUVS_INST_QUANTIZATION

#define CUVS_INST_VPQ_BUILD(T)                                                               \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::host_matrix_view<const T, int64_t, raft::row_major>& dataset)   \
  {                                                                                          \
    return detail::vpq_build_half<decltype(dataset)>(res, params, dataset);                  \
  }                                                                                          \
  auto vpq_build(const raft::resources& res,                                                 \
                 const cuvs::neighbors::vpq_params& params,                                  \
                 const raft::device_matrix_view<const T, int64_t, raft::row_major>& dataset) \
  {                                                                                          \
    return detail::vpq_build_half<decltype(dataset)>(res, params, dataset);                  \
  }

CUVS_INST_VPQ_BUILD(float);
CUVS_INST_VPQ_BUILD(half);
CUVS_INST_VPQ_BUILD(int8_t);
CUVS_INST_VPQ_BUILD(uint8_t);

#undef CUVS_INST_VPQ_BUILD

namespace {

template <typename T>
auto vpq_train_from_device_rows(raft::resources const& res,
                                cuvs::neighbors::vpq_params const& params,
                                T const* src_ptr,
                                int64_t n_rows,
                                int64_t dim,
                                int64_t stride) -> cuvs::neighbors::vpq_dataset<half, int64_t>
{
  auto stream = raft::resource::get_cuda_stream(res);
  if (stride != dim) {
    auto dense = raft::make_device_matrix<T, int64_t>(res, n_rows, dim);
    raft::copy_matrix(dense.data_handle(), dim, src_ptr, stride, dim, n_rows, stream);
    auto dense_view =
      raft::make_device_matrix_view<const T, int64_t>(dense.data_handle(), n_rows, dim);
    return vpq_build(res, params, dense_view);
  }
  auto row_view = raft::make_device_matrix_view<const T, int64_t>(src_ptr, n_rows, dim);
  return vpq_build(res, params, row_view);
}

}  // namespace

template <typename T>
cuvs::neighbors::vpq_dataset<half, int64_t> make_vpq_dataset(
  raft::resources const& res,
  cuvs::neighbors::vpq_params const& params,
  cuvs::neighbors::any_dataset_view<T, int64_t> const& dataset)
{
  using VT      = cuvs::neighbors::any_dataset_view_types<T, int64_t>;
  auto const& v = dataset.as_variant();
  if (std::holds_alternative<typename VT::empty_view>(v)) {
    RAFT_FAIL("make_vpq_dataset: dataset view is empty");
  }
  if (std::holds_alternative<typename VT::vpq_f16_view>(v) ||
      std::holds_alternative<typename VT::vpq_f32_view>(v)) {
    RAFT_FAIL(
      "make_vpq_dataset: source is already VPQ-compressed; train from dense device vectors");
  }
  if (std::holds_alternative<typename VT::padded_view>(v)) {
    auto const& padded = std::get<typename VT::padded_view>(v);
    const auto n_r     = static_cast<int64_t>(padded.n_rows());
    const auto d       = static_cast<int64_t>(padded.dim());
    const auto str     = static_cast<int64_t>(padded.stride());
    return vpq_train_from_device_rows<T>(res, params, padded.view().data_handle(), n_r, d, str);
  }
  if (std::holds_alternative<typename VT::strided_view>(v)) {
    auto const& strided = std::get<typename VT::strided_view>(v);
    const auto n_r      = static_cast<int64_t>(strided.n_rows());
    const auto d        = static_cast<int64_t>(strided.dim());
    const auto str      = static_cast<int64_t>(strided.stride());
    return vpq_train_from_device_rows<T>(res, params, strided.view().data_handle(), n_r, d, str);
  }
  RAFT_FAIL("make_vpq_dataset: unsupported dataset view alternative");
}

template cuvs::neighbors::vpq_dataset<half, int64_t> make_vpq_dataset<float>(
  raft::resources const&,
  cuvs::neighbors::vpq_params const&,
  cuvs::neighbors::any_dataset_view<float, int64_t> const&);
template cuvs::neighbors::vpq_dataset<half, int64_t> make_vpq_dataset<half>(
  raft::resources const&,
  cuvs::neighbors::vpq_params const&,
  cuvs::neighbors::any_dataset_view<half, int64_t> const&);
template cuvs::neighbors::vpq_dataset<half, int64_t> make_vpq_dataset<int8_t>(
  raft::resources const&,
  cuvs::neighbors::vpq_params const&,
  cuvs::neighbors::any_dataset_view<int8_t, int64_t> const&);
template cuvs::neighbors::vpq_dataset<half, int64_t> make_vpq_dataset<uint8_t>(
  raft::resources const&,
  cuvs::neighbors::vpq_params const&,
  cuvs::neighbors::any_dataset_view<uint8_t, int64_t> const&);

}  // namespace cuvs::preprocessing::quantize::pq
