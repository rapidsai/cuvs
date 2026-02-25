/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

#define INSTANTIATE_PREDICT(DataT, IndexT)                                      \
  template void predict<DataT, IndexT>(                                         \
    raft::resources const& handle,                                              \
    const kmeans::params& params,                                               \
    raft::device_matrix_view<const DataT, IndexT> X,                            \
    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<const DataT, IndexT> centroids,                    \
    raft::device_vector_view<IndexT, IndexT> labels,                            \
    bool normalize_weight,                                                      \
    raft::host_scalar_view<DataT> inertia);

INSTANTIATE_PREDICT(double, int)
INSTANTIATE_PREDICT(double, int64_t)

#undef INSTANTIATE_PREDICT

// --- Device-data predict ---

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
             raft::device_matrix_view<const double, int64_t> X,
             std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
             raft::device_matrix_view<const double, int64_t> centroids,
             raft::device_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia)
{
  cuvs::cluster::kmeans::predict<double, int64_t>(
    handle, params, X, sample_weight, centroids, labels, normalize_weight, inertia);
}

// --- Host-data predict ---

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::host_matrix_view<const double, int64_t> X,
             std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
             raft::device_matrix_view<const double, int64_t> centroids,
             raft::host_vector_view<int64_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia)
{
  auto batch_size = static_cast<int64_t>(params.batch_size > 0 ? params.batch_size : X.extent(0));
  cuvs::cluster::kmeans::predict<double, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, labels, normalize_weight, inertia);
}

}  // namespace cuvs::cluster::kmeans
