/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace cuvs::neighbors::epsilon_neighborhood {

/**
 * @defgroup epsilon_neighborhood_cpp_l2 Epsilon Neighborhood L2 Operations
 * @{
 */

/**
 * @brief Computes epsilon neighborhood for the L2-Squared distance metric and given ball size.
 * The epsilon neighbors is represented by a dense boolean adjacency matrix of size m * n and
 * an array of degrees for each vertex, which can be used as a compressed sparse row (CSR)
 * indptr array.
 *
 * @code{.cpp}
 *  #include <cuvs/neighbors/epsilon_neighborhood.hpp>
 *  #include <raft/core/resources.hpp>
 *  #include <raft/core/device_mdarray.hpp>
 *  using namespace cuvs::neighbors;
 *  raft::resources handle;
 *  ...
 *  auto x = raft::make_device_matrix<float, int64_t>(handle, m, k);
 *  auto y = raft::make_device_matrix<float, int64_t>(handle, n, k);
 *  auto adj = raft::make_device_matrix<bool, int64_t>(handle, m, n);
 *  auto vd = raft::make_device_vector<int, int64_t>(handle, m + 1);
 *  epsilon_neighborhood::eps_neighbors_l2sq(handle, x.view(), y.view(), adj.view(), vd.view(),
 * eps);
 * @endcode
 *
 * @tparam value_t   IO and math type
 * @tparam idx_t    Index type
 * @tparam matrix_idx_t matrix indexing type
 *
 * @param[in]  handle raft handle to manage library resources
 * @param[in]  x      first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  y      second matrix [row-major] [on device] [dim = n x k]
 * @param[out] adj    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  eps    defines epsilon neighborhood radius (should be passed as
 *                    squared as we compute L2-squared distance in this method)
 */
template <typename value_t, typename idx_t, typename matrix_idx_t>
void eps_neighbors_l2sq(raft::resources const& handle,
                        raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> x,
                        raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> y,
                        raft::device_matrix_view<bool, matrix_idx_t, raft::row_major> adj,
                        raft::device_vector_view<idx_t, matrix_idx_t> vd,
                        value_t eps);

/** @} */  // end group epsilon_neighborhood_cpp_l2

}  // namespace cuvs::neighbors::epsilon_neighborhood
