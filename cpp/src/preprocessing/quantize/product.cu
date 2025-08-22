/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "./detail/product.cuh"

#include <cuvs/preprocessing/quantize/product.hpp>

namespace cuvs::preprocessing::quantize::product {

#define CUVS_INST_QUANTIZATION(T, QuantI)                                         \
  auto train(raft::resources const& res,                                          \
             const params params,                                                 \
             raft::device_matrix_view<const T, int64_t> dataset) -> quantizer     \
  {                                                                               \
    return detail::train(res, params, dataset);                                   \
  }                                                                               \
  /*auto train(raft::resources const& res,                                        \
             const params params,                                                 \
             raft::host_matrix_view<const T, int64_t> dataset) -> quantizer       \
  {                                                                               \
    return detail::train(res, params, dataset);                                   \
  } */                                                                            \
  void transform(raft::resources const& res,                                      \
                 const quantizer& quantizer,                                      \
                 raft::device_matrix_view<const T, int64_t> dataset,              \
                 raft::device_matrix_view<QuantI, int64_t> out)                   \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  /*void transform(raft::resources const& res,                                    \
                 const quantizer& quantizer,                                      \
                 raft::host_matrix_view<const T, int64_t> dataset,                \
                 raft::host_matrix_view<QuantI, int64_t> out)                     \
  {                                                                               \
    detail::transform(res, quantizer, dataset, out);                              \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         const quantizer& quantizer,                              \
                         raft::device_matrix_view<const QuantI, int64_t> dataset, \
                         raft::device_matrix_view<T, int64_t> out)                \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  }                                                                               \
  void inverse_transform(raft::resources const& res,                              \
                         const quantizer& quantizer,                              \
                         raft::host_matrix_view<const QuantI, int64_t> dataset,   \
                         raft::host_matrix_view<T, int64_t> out)                  \
  {                                                                               \
    detail::inverse_transform(res, quantizer, dataset, out);                      \
  } */

auto train(
  raft::resources const& res,
  const params params,
  const uint32_t dim,
  raft::device_mdspan<const float, raft::extent_3d<uint32_t>, raft::row_major> pq_centers,
  raft::device_matrix_view<const float, uint32_t, raft::row_major> centers,
  std::optional<raft::device_matrix_view<const float, uint32_t, raft::row_major>> centers_rot,
  std::optional<raft::device_matrix_view<const float, uint32_t, raft::row_major>> rotation_matrix)
  -> quantizer
{
  return detail::train(res, params, dim, pq_centers, centers, centers_rot, rotation_matrix);
}

CUVS_INST_QUANTIZATION(float, uint8_t);
// CUVS_INST_QUANTIZATION(half, uint8_t);

#undef CUVS_INST_QUANTIZATION

}  // namespace cuvs::preprocessing::quantize::product
