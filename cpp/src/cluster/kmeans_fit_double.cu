/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

#define INSTANTIATE_FIT_MAIN(DataT, index_t)                       \
  template void fit_main<DataT, index_t>(                          \
    raft::resources const& handle,                                 \
    const kmeans::params& params,                                  \
    raft::device_matrix_view<const DataT, index_t> X,              \
    raft::device_vector_view<const DataT, index_t> sample_weights, \
    raft::device_matrix_view<DataT, index_t> centroids,            \
    raft::host_scalar_view<DataT> inertia,                         \
    raft::host_scalar_view<index_t> n_iter,                        \
    rmm::device_uvector<char>& workspace);

#define INSTANTIATE_FIT(DataT, index_t)                                          \
  template void fit<DataT, index_t>(                                             \
    raft::resources const& handle,                                               \
    const kmeans::params& params,                                                \
    raft::device_matrix_view<const DataT, index_t> X,                            \
    std::optional<raft::device_vector_view<const DataT, index_t>> sample_weight, \
    raft::device_matrix_view<DataT, index_t> centroids,                          \
    raft::host_scalar_view<DataT> inertia,                                       \
    raft::host_scalar_view<index_t> n_iter);

INSTANTIATE_FIT_MAIN(double, int)
INSTANTIATE_FIT_MAIN(double, int64_t)

INSTANTIATE_FIT(double, int)
INSTANTIATE_FIT(double, int64_t)

#undef INSTANTIATE_FIT_MAIN
#undef INSTANTIATE_FIT

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int> n_iter)
{
  cuvs::cluster::kmeans::fit<double, int>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const double, int64_t> X,
         std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::fit<double, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}
}  // namespace cuvs::cluster::kmeans
