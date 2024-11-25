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

#include <cuvs/neighbors/quantization.hpp>

namespace cuvs::neighbors::quantization {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                                         \
  auto scalar_quantize(raft::resources const& res,                                                \
                       params& params,                                                            \
                       raft::device_matrix_view<const T, int64_t> dataset)                        \
    ->raft::device_matrix<QuantI, int64_t>                                                        \
  {                                                                                               \
    return detail::scalar_quantize<T, QuantI>(res, params, dataset);                              \
  }                                                                                               \
  auto scalar_quantize(                                                                           \
    raft::resources const& res, params& params, raft::host_matrix_view<const T, int64_t> dataset) \
    ->raft::host_matrix<QuantI, int64_t>                                                          \
  {                                                                                               \
    return detail::scalar_quantize<T, QuantI>(res, params, dataset);                              \
  }

CUVS_INST_QUANTIZATION(double, int8_t);
CUVS_INST_QUANTIZATION(float, int8_t);
CUVS_INST_QUANTIZATION(half, int8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::neighbors::quantization