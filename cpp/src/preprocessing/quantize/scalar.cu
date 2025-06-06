/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "./detail/scalar.cuh"

#include <cuvs/preprocessing/quantize/scalar.hpp>

namespace cuvs::preprocessing::quantize::scalar {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                         \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer<T>  \
  {                                                                               \
    return detail::train(res, params, dataset);                                   \
  }                                                                               \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer<T>    \
  {                                                                               \
    return detail::train(res, params, dataset);                                   \
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
                 raft::host_matrix_view<QuantI, int64_t> out)                     \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         const quantizer<T>& quantizer,                           \
                         raft::device_matrix_view<const QuantI, int64_t> dataset, \
                         raft::device_matrix_view<T, int64_t> out)                \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         const quantizer<T>& quantizer,                           \
                         raft::host_matrix_view<const QuantI, int64_t> dataset,   \
                         raft::host_matrix_view<T, int64_t> out)                  \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  }                                                                               \
  template struct quantizer<T>;

CUVS_INST_QUANTIZATION(double, int8_t);
CUVS_INST_QUANTIZATION(float, int8_t);
CUVS_INST_QUANTIZATION(half, int8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::scalar
