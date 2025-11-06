/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

template void fit_main<double, int>(raft::resources const& handle,
                                    const kmeans::params& params,
                                    raft::device_matrix_view<const double, int> X,
                                    raft::device_vector_view<const double, int> sample_weights,
                                    raft::device_matrix_view<double, int> centroids,
                                    raft::host_scalar_view<double> inertia,
                                    raft::host_scalar_view<int> n_iter,
                                    rmm::device_uvector<char>& workspace);

template void fit_main<double, int64_t>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const double, int64_t> X,
  raft::device_vector_view<const double, int64_t> sample_weights,
  raft::device_matrix_view<double, int64_t> centroids,
  raft::host_scalar_view<double> inertia,
  raft::host_scalar_view<int64_t> n_iter,
  rmm::device_uvector<char>& workspace);

// Explicit instantiations (required because of extern template in header)
template void fit<double, int>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const double, int> X,
  std::optional<raft::device_vector_view<const double, int>> sample_weight,
  raft::device_matrix_view<double, int> centroids,
  raft::host_scalar_view<double> inertia,
  raft::host_scalar_view<int> n_iter);

template void fit<double, int64_t>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const double, int64_t> X,
  std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
  raft::device_matrix_view<double, int64_t> centroids,
  raft::host_scalar_view<double> inertia,
  raft::host_scalar_view<int64_t> n_iter);

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double, int> inertia,
         raft::host_scalar_view<int, int> n_iter)
{
  cuvs::cluster::kmeans::fit<double, int>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int64_t> X,
         std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double, int64_t> inertia,
         raft::host_scalar_view<int64_t, int64_t> n_iter)
{
  cuvs::cluster::kmeans::fit<double, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}
}  // namespace cuvs::cluster::kmeans
