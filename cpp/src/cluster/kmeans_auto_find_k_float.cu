/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kmeans.cuh"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::kmeans::helpers {

void find_k(raft::resources const& handle,
            raft::device_matrix_view<const float, int> X,
            raft::host_scalar_view<int> best_k,
            raft::host_scalar_view<float> inertia,
            raft::host_scalar_view<int> n_iter,
            int kmax,
            int kmin,
            int maxiter,
            float tol)
{
  cuvs::cluster::kmeans::find_k<int, float>(
    handle, X, best_k, inertia, n_iter, kmax, kmin, maxiter, tol);
}
}  // namespace cuvs::cluster::kmeans::helpers
