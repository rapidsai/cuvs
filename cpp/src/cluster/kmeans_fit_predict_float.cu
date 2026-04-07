/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans_impl_fit_predict.cuh"
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

namespace cuvs::cluster::kmeans {

void fit_predict(raft::resources const& handle,
                 const kmeans::params& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
                 std::optional<raft::device_matrix_view<float, int64_t>> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)
{
  auto n_samples    = X.extent(0);
  auto labels_int64 = raft::make_device_vector<int64_t, int64_t>(handle, n_samples);
  cuvs::cluster::kmeans::fit_predict<float, int64_t>(
    handle, params, X, sample_weight, centroids, labels_int64.view(), inertia, n_iter);
  raft::linalg::map(
    handle, labels, raft::cast_op<uint32_t>{}, raft::make_const_mdspan(labels_int64.view()));
}
}  // namespace cuvs::cluster::kmeans
