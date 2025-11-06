/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

// Explicit instantiations (required because of extern template in header)
template void predict<double, int>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const double, int> X,
  std::optional<raft::device_vector_view<const double, int>> sample_weight,
  raft::device_matrix_view<const double, int> centroids,
  raft::device_vector_view<int, int> labels,
  bool normalize_weight,
  raft::host_scalar_view<double> inertia);

template void predict<double, int64_t>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const double, int64_t> X,
  std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
  raft::device_matrix_view<const double, int64_t> centroids,
  raft::device_vector_view<int64_t, int64_t> labels,
  bool normalize_weight,
  raft::host_scalar_view<double> inertia);

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const double, int> X,
             std::optional<raft::device_vector_view<const double, int>> sample_weight,
             raft::device_matrix_view<const double, int> centroids,
             raft::device_vector_view<int, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia)

{
  cuvs::cluster::kmeans::predict<double, int>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const double, int> X,
             std::optional<raft::device_vector_view<const double, int>> sample_weight,
             raft::device_matrix_view<const double, int> centroids,
             raft::device_vector_view<int64_t, int> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia)

{
  cuvs::cluster::kmeans::predict<double, int64_t>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}
}  // namespace cuvs::cluster::kmeans
