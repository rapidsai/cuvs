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

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const double, int> X,
               raft::device_matrix_view<const double, int> centroids,
               raft::device_matrix_view<double, int> X_new)

{
  cuvs::cluster::kmeans::transform<double, int>(handle, params, X, centroids, X_new);
}
}  // namespace cuvs::cluster::kmeans
