/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
