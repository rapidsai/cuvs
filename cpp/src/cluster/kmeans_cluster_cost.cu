/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {
void cluster_cost(const raft::resources& handle,
                  raft::device_matrix_view<const float, int> X,
                  raft::device_matrix_view<const float, int> centroids,
                  raft::host_scalar_view<float> cost)
{
  cuvs::cluster::kmeans::cluster_cost<float, int>(handle, X, centroids, cost);
}

void cluster_cost(const raft::resources& handle,
                  raft::device_matrix_view<const double, int> X,
                  raft::device_matrix_view<const double, int> centroids,
                  raft::host_scalar_view<double> cost)
{
  cuvs::cluster::kmeans::cluster_cost<double, int>(handle, X, centroids, cost);
}
}  // namespace cuvs::cluster::kmeans
