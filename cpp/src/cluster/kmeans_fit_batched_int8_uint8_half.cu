/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../neighbors/detail/ann_utils.cuh"
#include "detail/kmeans_batched.cuh"
#include <raft/core/resources.hpp>

#include <cuda_fp16.h>

namespace cuvs::cluster::kmeans {

// Use the mapping struct from ann_utils for T -> float conversion
using cuvs::spatial::knn::detail::utils::mapping;

// Public API implementations - X is T (uint8/int8/half) but centroids are float
void fit_batched(raft::resources const& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::host_matrix_view<const uint8_t, int64_t> X,
                 int64_t batch_size,
                 std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::batched::detail::fit<uint8_t, float, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, inertia, n_iter, mapping<float>{});
}

void fit_batched(raft::resources const& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::host_matrix_view<const int8_t, int64_t> X,
                 int64_t batch_size,
                 std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::batched::detail::fit<int8_t, float, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, inertia, n_iter, mapping<float>{});
}

void fit_batched(raft::resources const& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::host_matrix_view<const half, int64_t> X,
                 int64_t batch_size,
                 std::optional<raft::host_vector_view<const float, int64_t>> sample_weight,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::host_scalar_view<float> inertia,
                 raft::host_scalar_view<int64_t> n_iter)
{
  cuvs::cluster::kmeans::batched::detail::fit<half, float, int64_t>(
    handle, params, X, batch_size, sample_weight, centroids, inertia, n_iter, mapping<float>{});
}

}  // namespace cuvs::cluster::kmeans
