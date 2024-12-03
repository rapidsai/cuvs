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

#include "./detail/quantization.cuh"

#include <cuvs/preprocessing/quantization.hpp>

namespace cuvs::preprocessing::quantization {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                           \
  template <>                                                                       \
  auto cuvs::preprocessing::quantization::train_scalar(                             \
    raft::resources const& res,                                                     \
    const cuvs::preprocessing::quantization::sq_params params,                      \
    raft::device_matrix_view<const T, int64_t> dataset)                             \
    ->cuvs::preprocessing::quantization::ScalarQuantizer<T, QuantI>                 \
  {                                                                                 \
    return detail::train_scalar<T, QuantI>(res, params, dataset);                   \
  }                                                                                 \
  template <>                                                                       \
  auto cuvs::preprocessing::quantization::train_scalar<T, QuantI>(                  \
    raft::resources const& res,                                                     \
    const cuvs::preprocessing::quantization::sq_params params,                      \
    raft::host_matrix_view<const T, int64_t> dataset)                               \
    ->cuvs::preprocessing::quantization::ScalarQuantizer<T, QuantI>                 \
  {                                                                                 \
    return detail::train_scalar<T, QuantI>(res, params, dataset);                   \
  }                                                                                 \
  template <>                                                                       \
  void cuvs::preprocessing::quantization::transform(                                \
    raft::resources const& res,                                                     \
    const cuvs::preprocessing::quantization::ScalarQuantizer<T, QuantI>& quantizer, \
    raft::device_matrix_view<const T, int64_t> dataset,                             \
    raft::device_matrix_view<QuantI, int64_t> out)                                  \
  {                                                                                 \
    detail::transform<T, QuantI>(res, quantizer, dataset, out);                     \
  }                                                                                 \
  template <>                                                                       \
  void cuvs::preprocessing::quantization::transform(                                \
    raft::resources const& res,                                                     \
    const cuvs::preprocessing::quantization::ScalarQuantizer<T, QuantI>& quantizer, \
    raft::host_matrix_view<const T, int64_t> dataset,                               \
    raft::host_matrix_view<QuantI, int64_t> out)                                    \
  {                                                                                 \
    detail::transform<T, QuantI>(res, quantizer, dataset, out);                     \
  }                                                                                 \
  template <>                                                                       \
  void cuvs::preprocessing::quantization::inverse_transform(                        \
    raft::resources const& res,                                                     \
    const cuvs::preprocessing::quantization::ScalarQuantizer<T, QuantI>& quantizer, \
    raft::device_matrix_view<const QuantI, int64_t> dataset,                        \
    raft::device_matrix_view<T, int64_t> out)                                       \
  {                                                                                 \
    detail::inverse_transform<T, QuantI>(res, quantizer, dataset, out);             \
  }                                                                                 \
  template <>                                                                       \
  void cuvs::preprocessing::quantization::inverse_transform(                        \
    raft::resources const& res,                                                     \
    const cuvs::preprocessing::quantization::ScalarQuantizer<T, QuantI>& quantizer, \
    raft::host_matrix_view<const QuantI, int64_t> dataset,                          \
    raft::host_matrix_view<T, int64_t> out)                                         \
  {                                                                                 \
    detail::inverse_transform<T, QuantI>(res, quantizer, dataset, out);             \
  }                                                                                 \
  template struct cuvs::preprocessing::quantization::ScalarQuantizer<T, QuantI>;

CUVS_INST_QUANTIZATION(double, int8_t);
CUVS_INST_QUANTIZATION(float, int8_t);
CUVS_INST_QUANTIZATION(half, int8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantization