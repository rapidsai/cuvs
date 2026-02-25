/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

#define INSTANTIATE_FIT_MAIN(DataT, IndexT)                       \
  template void fit_main<DataT, IndexT>(                          \
    raft::resources const& handle,                                \
    const kmeans::params& params,                                 \
    raft::device_matrix_view<const DataT, IndexT> X,              \
    raft::device_vector_view<const DataT, IndexT> sample_weights, \
    raft::device_matrix_view<DataT, IndexT> centroids,            \
    raft::host_scalar_view<DataT> inertia,                        \
    raft::host_scalar_view<IndexT> n_iter,                        \
    rmm::device_uvector<char>& workspace);

#define INSTANTIATE_FIT(DataT, IndexT)                                          \
  template void fit<DataT, IndexT>(                                             \
    raft::resources const& handle,                                              \
    const kmeans::params& params,                                               \
    raft::device_matrix_view<const DataT, IndexT> X,                            \
    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<DataT, IndexT> centroids,                          \
    raft::host_scalar_view<DataT> inertia,                                      \
    raft::host_scalar_view<IndexT> n_iter);

INSTANTIATE_FIT_MAIN(double, int)
INSTANTIATE_FIT_MAIN(double, int64_t)

INSTANTIATE_FIT(double, int)
INSTANTIATE_FIT(double, int64_t)

#undef INSTANTIATE_FIT_MAIN
#undef INSTANTIATE_FIT

// --- Device-data fit ---

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

// --- Host-data fit ---

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const double, int64_t> X,
         std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
         raft::device_matrix_view<double, int64_t> centroids,
         raft::host_scalar_view<double> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  auto batch_size = static_cast<int64_t>(params.batch_size > 0 ? params.batch_size : X.extent(0));
  cuvs::cluster::kmeans::fit<double, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, inertia, n_iter);
}

// --- Mini-batch fit ---

void minibatch_fit(raft::resources const& handle,
                   const cuvs::cluster::kmeans::params& params,
                   raft::host_matrix_view<const double, int64_t> X,
                   std::optional<raft::host_vector_view<const double, int64_t>> sample_weight,
                   raft::device_matrix_view<double, int64_t> centroids,
                   raft::host_scalar_view<double> inertia,
                   raft::host_scalar_view<int64_t> n_iter)
{
  auto batch_size = static_cast<int64_t>(params.batch_size > 0 ? params.batch_size : X.extent(0));
  cuvs::cluster::kmeans::minibatch_fit<double, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans
