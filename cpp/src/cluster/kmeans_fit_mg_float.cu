/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/kmeans_mg.cuh"
#include "kmeans_mg.hpp"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans::mg {

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int> n_iter,
         std::optional<raft::device_vector_view<int, int>> labels)
{
  rmm::device_uvector<char> workspace(0, raft::resource::get_cuda_stream(handle));

  cuvs::cluster::kmeans::mg::detail::fit<float, int>(
    handle, params, X, sample_weight, centroids, inertia, n_iter, workspace, labels);
}

void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const float, int64_t> X,
         std::optional<raft::device_vector_view<const float, int64_t>> sample_weight,
         raft::device_matrix_view<float, int64_t> centroids,
         raft::host_scalar_view<float> inertia,
         raft::host_scalar_view<int64_t> n_iter,
         std::optional<raft::device_vector_view<int64_t, int64_t>> labels)
{
  rmm::device_uvector<char> workspace(0, raft::resource::get_cuda_stream(handle));

  cuvs::cluster::kmeans::mg::detail::fit<float, int64_t>(
    handle, params, X, sample_weight, centroids, inertia, n_iter, workspace, labels);
}
}  // namespace cuvs::cluster::kmeans::mg
