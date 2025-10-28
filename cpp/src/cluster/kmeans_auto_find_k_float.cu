/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kmeans.cuh"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans::helpers {

void find_k(raft::resources const& handle,
            raft::device_matrix_view<const float, int> X,
            raft::host_scalar_view<int, int> best_k,
            raft::host_scalar_view<float, int> inertia,
            raft::host_scalar_view<int, int> n_iter,
            int kmax,
            int kmin,
            int maxiter,
            float tol)
{
  cuvs::cluster::kmeans::find_k<int, float>(
    handle, X, best_k, inertia, n_iter, kmax, kmin, maxiter, tol);
}
}  // namespace cuvs::cluster::kmeans::helpers
