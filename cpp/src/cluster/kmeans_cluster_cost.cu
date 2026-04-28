/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void cluster_cost(const raft::resources& handle,
                  raft::device_matrix_view<const float, int64_t> X,
                  raft::device_matrix_view<const float, int64_t> centroids,
                  raft::host_scalar_view<float> cost,
                  std::optional<raft::device_vector_view<const float, int64_t>> sample_weight)
{
  cuvs::cluster::kmeans::cluster_cost<float, int64_t>(handle, X, centroids, cost, sample_weight);
}

void cluster_cost(const raft::resources& handle,
                  raft::device_matrix_view<const double, int64_t> X,
                  raft::device_matrix_view<const double, int64_t> centroids,
                  raft::host_scalar_view<double> cost,
                  std::optional<raft::device_vector_view<const double, int64_t>> sample_weight)
{
  cuvs::cluster::kmeans::cluster_cost<double, int64_t>(handle, X, centroids, cost, sample_weight);
}
}  // namespace cuvs::cluster::kmeans
