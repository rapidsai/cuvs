/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl_fit_predict.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

// --- Device-data fit_predict ---

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

// --- Host-data fit_predict ---

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::host_matrix_view<const double, int64_t> X,
                 std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
                 raft::device_matrix_view<double, int64_t> centroids,
                 raft::host_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<double> inertia,
                 raft::host_scalar_view<int64_t> n_iter)
{
  auto batch_size = static_cast<int64_t>(params.batch_size > 0 ? params.batch_size : X.extent(0));
  cuvs::cluster::kmeans::fit_predict<double, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, labels, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
