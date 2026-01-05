/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/pq.cuh"

#include <cuvs/preprocessing/quantize/pq.hpp>

namespace cuvs::preprocessing::quantize::pq {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                         \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>  \
  {                                                                               \
    return detail::train<T, T>(res, params, dataset);                             \
  }                                                                               \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>    \
  {                                                                               \
    return detail::train<T, T>(res, params, dataset);                             \
  }                                                                               \
  void transform(raft::resources const& res,                                      \
                 const quantizer<T>& quantizer,                                   \
                 raft::device_matrix_view<const T, int64_t> dataset,              \
                 raft::device_matrix_view<QuantI, int64_t> out)                   \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void transform(raft::resources const& res,                                      \
                 const quantizer<T>& quantizer,                                   \
                 raft::host_matrix_view<const T, int64_t> dataset,                \
                 raft::device_matrix_view<QuantI, int64_t> out)                   \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         const quantizer<T>& quantizer,                           \
                         raft::device_matrix_view<const QuantI, int64_t> dataset, \
                         raft::device_matrix_view<T, int64_t> out)                \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  }

CUVS_INST_QUANTIZATION(float, uint8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::pq
