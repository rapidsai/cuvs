/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

template <typename DataT, typename IndexT>
void fit_predict(raft::resources const& handle,
                 const kmeans::params& pams,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                 std::optional<raft::device_matrix_view<DataT, IndexT>> centroids,
                 raft::device_vector_view<IndexT, IndexT> labels,
                 raft::host_scalar_view<DataT> inertia,
                 raft::host_scalar_view<IndexT> n_iter)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeans_fit_predict");
  if (!centroids.has_value()) {
    auto n_features = X.extent(1);
    auto centroids_matrix =
      raft::make_device_matrix<DataT, IndexT>(handle, pams.n_clusters, n_features);
    cuvs::cluster::kmeans::fit<DataT, IndexT>(
      handle, pams, X, sample_weight, centroids_matrix.view(), inertia, n_iter);
    cuvs::cluster::kmeans::predict<DataT, IndexT>(
      handle, pams, X, sample_weight, centroids_matrix.view(), labels, true, inertia);
  } else {
    cuvs::cluster::kmeans::fit<DataT, IndexT>(
      handle, pams, X, sample_weight, centroids.value(), inertia, n_iter);
    cuvs::cluster::kmeans::predict<DataT, IndexT>(
      handle, pams, X, sample_weight, centroids.value(), labels, true, inertia);
  }
}

// Explicit instantiations (required because of extern template in header)
template void fit_predict<float, int>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const float, int> X,
  std::optional<raft::device_vector_view<const float, int>> sample_weight,
  std::optional<raft::device_matrix_view<float, int>> centroids,
  raft::device_vector_view<int, int> labels,
  raft::host_scalar_view<float> inertia,
  raft::host_scalar_view<int> n_iter);

template void fit_predict<float, int64_t>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const float, int64_t> X,
  std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
  std::optional<raft::device_matrix_view<float, int64_t>> centroids,
  raft::device_vector_view<int64_t, int64_t> labels,
  raft::host_scalar_view<float> inertia,
  raft::host_scalar_view<int64_t> n_iter);

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int> X,
                 std::optional<raft::device_vector_view<const float, int>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int>> centroids,
                 raft::device_vector_view<int, int> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int> n_iter)

{
  cuvs::cluster::kmeans::fit_predict<float, int>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int64_t>> centroids,
                 raft::device_vector_view<int64_t, int64_t> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)

{
  cuvs::cluster::kmeans::fit_predict<float, int64_t>(
    handle, params, X, sample_weight, centroids, labels, inertia, n_iter);
}
}  // namespace cuvs::cluster::kmeans
