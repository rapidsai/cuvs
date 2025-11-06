/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

template <typename DataT, typename IndexT>
void fit_main(raft::resources const& handle,
              const kmeans::params& params,
              raft::device_matrix_view<const DataT, IndexT> X,
              raft::device_vector_view<const DataT, IndexT> sample_weights,
              raft::device_matrix_view<DataT, IndexT> centroids,
              raft::host_scalar_view<DataT> inertia,
              raft::host_scalar_view<IndexT> n_iter,
              rmm::device_uvector<char>& workspace)
{
  cuvs::cluster::kmeans::detail::kmeans_fit_main<DataT, IndexT>(
    handle, params, X, sample_weights, centroids, inertia, n_iter, workspace);
}

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

template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const kmeans::params& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  // use the mnmg kmeans fit if we have comms initialize, single gpu otherwise
  if (raft::resource::comms_initialized(handle)) {
    cuvs::cluster::kmeans::mg::fit(handle, params, X, sample_weight, centroids, inertia, n_iter);
  } else {
    cuvs::cluster::kmeans::detail::kmeans_fit<DataT, IndexT>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
  }
}

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
