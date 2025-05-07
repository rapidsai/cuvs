/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int> X,
                 std::optional<raft::device_vector_view<const float, int>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int>> centroids,
                 raft::device_vector_view<int, int> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int> n_iter)

{
  cuvs::cluster::kmeans::fit_predict<float, int>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)

{
  cuvs::cluster::kmeans::fit_predict<float, int64_t>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}
}  // namespace cuvs::cluster::kmeans
