/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

// clang-format off
#include "kmeans_balanced.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace cuvs::cluster::kmeans {

void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const int8_t, int> X,
                 raft::device_matrix_view<float, int> centroids,
                 raft::device_vector_view<uint32_t, int> labels)
{
  cuvs::cluster::kmeans_balanced::fit_predict(
    handle, params, X, centroids, labels, cuvs::spatial::knn::detail::utils::mapping<float>{});
}

void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const int8_t, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels)
{
  cuvs::cluster::kmeans_balanced::fit_predict(
    handle, params, X, centroids, labels, cuvs::spatial::knn::detail::utils::mapping<float>{});
}
}  // namespace cuvs::cluster::kmeans
