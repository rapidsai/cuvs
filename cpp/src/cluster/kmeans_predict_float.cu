/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

template <typename DataT, typename IndexT>
void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const DataT, IndexT> X,
             std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
             raft::device_matrix_view<const DataT, IndexT> centroids,
             raft::device_vector_view<IndexT, IndexT> labels,
             bool normalize_weight,
             raft::host_scalar_view<DataT> inertia)
{
  cuvs::cluster::kmeans::detail::kmeans_predict<DataT, IndexT>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

// Explicit instantiations (required because of extern template in header)
template void predict<float, int>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const float, int> X,
  std::optional<raft::device_vector_view<const float, int>> sample_weight,
  raft::device_matrix_view<const float, int> centroids,
  raft::device_vector_view<int, int> labels,
  bool normalize_weight,
  raft::host_scalar_view<float> inertia);

template void predict<float, int64_t>(
  raft::resources const& handle,
  const kmeans::params& params,
  raft::device_matrix_view<const float, int64_t> X,
  std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
  raft::device_matrix_view<const float, int64_t> centroids,
  raft::device_vector_view<int64_t, int64_t> labels,
  bool normalize_weight,
  raft::host_scalar_view<float> inertia);

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
