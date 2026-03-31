/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// clang-format off
#include "kmeans_balanced.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace cuvs::cluster::kmeans::helpers {

void build_clusters(const raft::resources& handle,
                    cuvs::cluster::kmeans::balanced_params const& params,
                    raft::device_matrix_view<const float, int64_t> X,
                    raft::device_matrix_view<float, int64_t> centroids,
                    raft::device_vector_view<uint32_t, int64_t> labels,
                    raft::device_vector_view<uint32_t, int64_t> cluster_sizes)
{
  cuvs::cluster::kmeans_balanced::helpers::build_clusters(
    handle,
    params,
    X,
    centroids,
    labels,
    cluster_sizes,
    cuvs::spatial::knn::detail::utils::mapping<float>{},
    std::optional<raft::device_vector_view<const float>>{std::nullopt});
}

}  // namespace cuvs::cluster::kmeans::helpers
