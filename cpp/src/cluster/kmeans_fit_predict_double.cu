/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const double, int> X,
                 std::optional<raft::device_vector_view<const double, int>> sample_weight,
                 std::optional<raft::device_matrix_view<double, int>> centroids,
                 raft::device_vector_view<int, int> labels,
                 raft::host_scalar_view<double> inertia,
                 raft::host_scalar_view<int> n_iter)

{
  cuvs::cluster::kmeans::fit_predict<double, int>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const double, int64_t> X,
                 std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<double, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<double> inertia,
                 raft::host_scalar_view<int64_t> n_iter)

{
  cuvs::cluster::kmeans::fit_predict<double, int64_t>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}
}  // namespace cuvs::cluster::kmeans
