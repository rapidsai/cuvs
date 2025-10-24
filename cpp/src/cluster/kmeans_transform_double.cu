/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans {

void transform(raft::resources const& handle,
               const kmeans::params& params,
               raft::device_matrix_view<const double, int> X,
               raft::device_matrix_view<const double, int> centroids,
               raft::device_matrix_view<double, int> X_new)

{
  cuvs::cluster::kmeans::transform<double, int>(handle, params, X, centroids, X_new);
}
}  // namespace cuvs::cluster::kmeans
