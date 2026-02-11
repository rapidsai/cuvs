/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

#define INSTANTIATE_PREDICT(DataT, index_t)                                      \
  template void predict<DataT, index_t>(                                         \
    raft::resources const& handle,                                               \
    const kmeans::params& params,                                                \
    raft::device_matrix_view<const DataT, index_t> X,                            \
    std::optional<raft::device_vector_view<const DataT, index_t>> sample_weight, \
    raft::device_matrix_view<const DataT, index_t> centroids,                    \
    raft::device_vector_view<index_t, index_t> labels,                           \
    bool normalize_weight,                                                       \
    raft::host_scalar_view<DataT> inertia);

INSTANTIATE_PREDICT(double, int)
INSTANTIATE_PREDICT(double, int64_t)

#undef INSTANTIATE_PREDICT

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
}  // namespace cuvs::cluster::kmeans
