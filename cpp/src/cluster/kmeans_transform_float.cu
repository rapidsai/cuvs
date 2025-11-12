/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const float, int> X,
               raft::device_matrix_view<const float, int> centroids,
               raft::device_matrix_view<float, int> X_new)

{
  cuvs::cluster::kmeans::transform<float, int>(handle, params, X, centroids, X_new);
}
}  // namespace cuvs::cluster::kmeans
