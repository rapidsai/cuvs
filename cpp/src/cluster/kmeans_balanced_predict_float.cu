/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// clang-format off
#include "kmeans_balanced.cuh"
#include "kmeans_balanced_build_clusters_impl.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace cuvs::cluster::kmeans {

void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const float, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<uint32_t, int64_t> labels)
{
  cuvs::cluster::kmeans_balanced::predict(
    handle, params, X, centroids, labels, cuvs::spatial::knn::detail::utils::mapping<float>{});
}

void predict(const raft::resources& handle,
             cuvs::cluster::kmeans::balanced_params const& params,
             raft::device_matrix_view<const float, int64_t> X,
             raft::device_matrix_view<const float, int64_t> centroids,
             raft::device_vector_view<int, int64_t> labels)
{
  cuvs::cluster::kmeans_balanced::predict(
    handle, params, X, centroids, labels, cuvs::spatial::knn::detail::utils::mapping<float>{});
}
}  // namespace cuvs::cluster::kmeans

namespace cuvs::cluster::kmeans_balanced::helpers {

#define INSTANTIATE_BUILD_CLUSTERS(DataT, MathT, index_t, label_t, CounterT, MappingOpT) \
  template void build_clusters<DataT, MathT, index_t, label_t, CounterT, MappingOpT>(    \
    const raft::resources& handle,                                                       \
    const cuvs::cluster::kmeans::balanced_params& params,                                \
    raft::device_matrix_view<const DataT, index_t> X,                                    \
    raft::device_matrix_view<MathT, index_t> centroids,                                  \
    raft::device_vector_view<label_t, index_t> labels,                                   \
    raft::device_vector_view<CounterT, index_t> cluster_sizes,                           \
    MappingOpT mapping_op,                                                               \
    std::optional<raft::device_vector_view<const MathT>> X_norm);

// Explicit instantiation for the build_clusters actually used in IVF-PQ build
// Placed here because predict shares similar code paths with build_clusters
INSTANTIATE_BUILD_CLUSTERS(
  float, float, int64_t, uint32_t, uint32_t, cuvs::spatial::knn::detail::utils::mapping<float>)

#undef INSTANTIATE_BUILD_CLUSTERS

}  // namespace cuvs::cluster::kmeans_balanced::helpers
