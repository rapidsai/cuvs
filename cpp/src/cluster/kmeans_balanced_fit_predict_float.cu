/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// clang-format off
#include "kmeans_balanced_impl_fit_predict.cuh"
#include "../neighbors/detail/ann_utils.cuh"
#include <raft/core/resources.hpp>
// clang-format on

namespace cuvs::cluster::kmeans {

void fit_predict(const raft::resources& handle,
                 cuvs::cluster::kmeans::balanced_params const& params,
                 raft::device_matrix_view<const float, int64_t> X,
                 raft::device_matrix_view<float, int64_t> centroids,
                 raft::device_vector_view<uint32_t, int64_t> labels)
{
  cuvs::cluster::kmeans_balanced::fit_predict(handle, params, X, centroids, labels);
}
}  // namespace cuvs::cluster::kmeans
