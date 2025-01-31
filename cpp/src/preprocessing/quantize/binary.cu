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

#include "./detail/binary.cuh"

#include <cuvs/preprocessing/quantize/binary.hpp>

namespace cuvs::preprocessing::quantize::binary {

#define CUVS_INST_QUANTIZATION(T, QuantI)                            \
  void transform(raft::resources const& res,                         \
                 raft::device_matrix_view<const T, int64_t> dataset, \
                 raft::device_matrix_view<QuantI, int64_t> out)      \
  {                                                                  \
    detail::transform(res, dataset, out);                            \
  }                                                                  \
  void transform(raft::resources const& res,                         \
                 raft::host_matrix_view<const T, int64_t> dataset,   \
                 raft::host_matrix_view<QuantI, int64_t> out)        \
  {                                                                  \
    detail::transform(res, dataset, out);                            \
  }

CUVS_INST_QUANTIZATION(double, uint8_t);
CUVS_INST_QUANTIZATION(float, uint8_t);
CUVS_INST_QUANTIZATION(half, uint8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::binary
