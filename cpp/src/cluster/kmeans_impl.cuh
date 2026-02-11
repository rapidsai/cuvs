/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"

namespace cuvs::cluster::kmeans {

template <typename DataT, typename index_t>
void fit_main(raft::resources const& handle,
              const kmeans::params& params,
              raft::device_matrix_view<const DataT, index_t> X,
              raft::device_vector_view<const DataT, index_t> sample_weights,
              raft::device_matrix_view<DataT, index_t> centroids,
              raft::host_scalar_view<DataT> inertia,
              raft::host_scalar_view<index_t> n_iter,
              rmm::device_uvector<char>& workspace)
{
  cuvs::cluster::kmeans::detail::kmeans_fit_main<DataT, index_t>(
    handle, params, X, sample_weights, centroids, inertia, n_iter, workspace);
}

template <typename DataT, typename index_t>
void fit(raft::resources const& handle,
         const kmeans::params& params,
         raft::device_matrix_view<const DataT, index_t> X,
         std::optional<raft::device_vector_view<const DataT, index_t>> sample_weight,
         raft::device_matrix_view<DataT, index_t> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<index_t> n_iter)
{
  // use the mnmg kmeans fit if we have comms initialize, single gpu otherwise
  if (raft::resource::comms_initialized(handle)) {
    cuvs::cluster::kmeans::mg::fit(handle, params, X, sample_weight, centroids, inertia, n_iter);
  } else {
    cuvs::cluster::kmeans::detail::kmeans_fit<DataT, index_t>(
      handle, params, X, sample_weight, centroids, inertia, n_iter);
  }
}

template <typename DataT, typename index_t>
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const DataT, index_t> X,
             std::optional<raft::device_vector_view<const DataT, index_t>> sample_weight,
             raft::device_matrix_view<const DataT, index_t> centroids,
             raft::device_vector_view<index_t, index_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<DataT> inertia)
{
  cuvs::cluster::kmeans::detail::kmeans_predict<DataT, index_t>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

}  // namespace cuvs::cluster::kmeans
