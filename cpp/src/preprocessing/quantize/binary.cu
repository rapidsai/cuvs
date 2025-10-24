/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/binary.cuh"

#include <cuvs/preprocessing/quantize/binary.hpp>

namespace cuvs::preprocessing::quantize::binary {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                                \
  auto train(raft::resources const& res,                                                 \
             const params params,                                                        \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>           \
  {                                                                                      \
    return detail::train(res, params, dataset);                                          \
  }                                                                                      \
  auto train(raft::resources const& res,                                                 \
             const params params,                                                        \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>         \
  {                                                                                      \
    return detail::train(res, params, dataset);                                          \
  }                                                                                      \
  void transform(raft::resources const& res,                                             \
                 const cuvs::preprocessing::quantize::binary::quantizer<T>& quantizer,   \
                 raft::device_matrix_view<const T, int64_t> dataset,                     \
                 raft::device_matrix_view<QuantI, int64_t> out)                          \
  {                                                                                      \
    detail::transform(res, quantizer, dataset, out);                                     \
  }                                                                                      \
  void transform(raft::resources const& res,                                             \
                 const cuvs::preprocessing::quantize::binary::quantizer<T>& quantizer,   \
                 raft::host_matrix_view<const T, int64_t> dataset,                       \
                 raft::host_matrix_view<QuantI, int64_t> out)                            \
  {                                                                                      \
    detail::transform(res, quantizer, dataset, out);                                     \
  }                                                                                      \
  void transform(raft::resources const& res,                                             \
                 raft::device_matrix_view<const T, int64_t> dataset,                     \
                 raft::device_matrix_view<QuantI, int64_t> out)                          \
  {                                                                                      \
    cuvs::preprocessing::quantize::binary::params params{                                \
      .threshold = cuvs::preprocessing::quantize::binary::bit_threshold::zero};          \
    auto quantizer = cuvs::preprocessing::quantize::binary::train(res, params, dataset); \
    detail::transform(res, quantizer, dataset, out);                                     \
  }                                                                                      \
  void transform(raft::resources const& res,                                             \
                 raft::host_matrix_view<const T, int64_t> dataset,                       \
                 raft::host_matrix_view<QuantI, int64_t> out)                            \
  {                                                                                      \
    cuvs::preprocessing::quantize::binary::params params{                                \
      .threshold = cuvs::preprocessing::quantize::binary::bit_threshold::zero};          \
    auto quantizer = cuvs::preprocessing::quantize::binary::train(res, params, dataset); \
    detail::transform(res, quantizer, dataset, out);                                     \
  }

CUVS_INST_QUANTIZATION(double, uint8_t);
CUVS_INST_QUANTIZATION(float, uint8_t);
CUVS_INST_QUANTIZATION(half, uint8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::binary
