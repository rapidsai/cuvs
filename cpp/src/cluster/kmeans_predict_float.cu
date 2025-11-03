/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int> X,
             std::optional<raft::device_vector_view<const float, int>> sample_weight,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<int, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)

{
  cuvs::cluster::kmeans::predict<float, int>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const float, int> X,
             std::optional<raft::device_vector_view<const float, int>> sample_weight,
             raft::device_matrix_view<const float, int> centroids,
             raft::device_vector_view<int64_t, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<float> inertia)

{
  cuvs::cluster::kmeans::predict<float, int64_t>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}
}  // namespace cuvs::cluster::kmeans
