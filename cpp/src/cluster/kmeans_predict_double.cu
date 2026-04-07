/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include "kmeans_impl.cuh"
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

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

INSTANTIATE_PREDICT(double, int64_t)

#undef INSTANTIATE_PREDICT

void predict(raft::resources const& handle,
             const kmeans::params& params,
             raft::device_matrix_view<const double, int64_t> X,
             std::optional<raft::device_vector_view<const double, int64_t>> sample_weight,
             raft::device_matrix_view<const double, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels,
             bool normalize_weight,
             raft::host_scalar_view<double> inertia)
{
  auto n_samples    = X.extent(0);
  auto labels_int64 = raft::make_device_vector<int64_t, int64_t>(handle, n_samples);
  cuvs::cluster::kmeans::predict<double, int64_t>(
    handle, params, X, sample_weight, centroids, labels_int64.view(), normalize_weight, inertia);
  raft::linalg::map(
    handle, labels, raft::cast_op<uint32_t>{}, raft::make_const_mdspan(labels_int64.view()));
}
}  // namespace cuvs::cluster::kmeans
