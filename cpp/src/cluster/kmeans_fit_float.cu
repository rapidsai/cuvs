/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/kmeans_batched.cuh"
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

#define INSTANTIATE_FIT_BATCHED(DataT, IndexT)                                \
  template void batched::detail::fit<DataT, IndexT>(                          \
    raft::resources const& handle,                                            \
    const kmeans::params& params,                                             \
    raft::host_matrix_view<const DataT, IndexT> X,                            \
    IndexT batch_size,                                                        \
    std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight, \
    raft::device_matrix_view<DataT, IndexT> centroids,                        \
    raft::host_scalar_view<DataT> inertia,                                    \
    raft::host_scalar_view<IndexT> n_iter);

INSTANTIATE_FIT_MAIN(float, int)
INSTANTIATE_FIT_MAIN(float, int64_t)

INSTANTIATE_FIT(float, int)
INSTANTIATE_FIT(float, int64_t)

INSTANTIATE_FIT_BATCHED(float, int)
INSTANTIATE_FIT_BATCHED(float, int64_t)

#undef INSTANTIATE_FIT_MAIN
#undef INSTANTIATE_FIT
#undef INSTANTIATE_FIT_BATCHED

void fit_batched(raft::resources const& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::host_matrix_view<const float, int> X,
                 int batch_size,
                 std::optional<raft::host_vector_view<const float, int>> sample_weight,
                 raft::device_matrix_view<float, int> centroids,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int> n_iter)
{
  cuvs::cluster::kmeans::batched::detail::fit<float, int>(
    handle, params, X, batch_size, sample_weight, centroids, inertia, n_iter);
}

void fit_batched(raft::resources const& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::host_matrix_view<const float, int64_t> X,
                 int64_t batch_size,
                 std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::batched::detail::fit<float, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int> n_iter)
{
  cuvs::cluster::kmeans::fit<float, int>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::fit<float, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}
}  // namespace cuvs::cluster::kmeans
