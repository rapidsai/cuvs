/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// clang-format off
#include "kmeans_balanced.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace cuvs::cluster::kmeans {

void fit(const raft::resources& handle,
         cuvs::cluster::kmeans::balanced_params const& params,
         raft::device_matrix_view<const uint8_t, int64_t> X,
         raft::device_matrix_view<float, int64_t> centroids,
         std::optional<raft::host_scalar_view<float>> inertia)
{
  cuvs::cluster::kmeans_balanced::fit(
    handle, params, X, centroids, cuvs::spatial::knn::detail::utils::mapping<float>{}, inertia);
}

void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const uint8_t, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels,
                 std::optional<raft::host_scalar_view<float>> inertia)
{
  cuvs::cluster::kmeans_balanced::fit_predict(handle,
                                              params,
                                              X,
                                              centroids,
                                              labels,
                                              cuvs::spatial::knn::detail::utils::mapping<float>{},
                                              inertia);
}
}  // namespace cuvs::cluster::kmeans
