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

#include "./detail/kmeans_mg.cuh"
#include "kmeans_mg.hpp"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans::mg {

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float, int> inertia,
         raft::host_scalar_view<int, int> n_iter)
{
  rmm::device_uvector<char> workspace(0, raft::resource::get_cuda_stream(handle));

  cuvs::cluster::kmeans::mg::detail::fit<float, int>(
    handle, params, X, sample_weight.value(), centroids, inertia, n_iter, workspace);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float, int64_t> inertia,
         raft::host_scalar_view<int64_t, int64_t> n_iter)
{
  rmm::device_uvector<char> workspace(0, raft::resource::get_cuda_stream(handle));

  cuvs::cluster::kmeans::mg::detail::fit<float, int64_t>(
    handle, params, X, sample_weight.value(), centroids, inertia, n_iter, workspace);
}
}  // namespace cuvs::cluster::kmeans::mg
