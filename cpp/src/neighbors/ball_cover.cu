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

#include "ball_cover.cuh"
#include <cuvs/neighbors/ball_cover.hpp>

namespace cuvs::neighbors::ball_cover {

void build(raft::resources const& handle, cuvs::neighbors::ball_cover::index<int64_t, float>& index)
{
  detail::build_index<int64_t, float>(handle, index);
}

void all_knn_query(raft::resources const& handle,
                   cuvs::neighbors::ball_cover::index<int64_t, float>& index,
                   raft::device_matrix_view<int64_t, int64_t, raft::row_major> inds,
                   raft::device_matrix_view<float, int64_t, raft::row_major> dists,
                   bool perform_post_filtering,
                   float weight)
{
  detail::all_knn_query<int64_t, float>(handle, index, inds, dists, perform_post_filtering, weight);
}

void eps_nn(raft::resources const& handle,
            const cuvs::neighbors::ball_cover::index<int64_t, float>& index,
            raft::device_matrix_view<bool, int64_t, raft::row_major> adj,
            raft::device_vector_view<int64_t, int64_t> vd,
            raft::device_matrix_view<const float, int64_t, raft::row_major> query,
            float eps)
{
  detail::eps_nn<int64_t, float>(handle, index, adj, vd, query, eps);
}

void eps_nn(raft::resources const& handle,
            const cuvs::neighbors::ball_cover::index<int64_t, float>& index,
            raft::device_vector_view<int64_t, int64_t> adj_ia,
            raft::device_vector_view<int64_t, int64_t> adj_ja,
            raft::device_vector_view<int64_t, int64_t> vd,
            raft::device_matrix_view<const float, int64_t, raft::row_major> query,
            float eps,
            std::optional<raft::host_scalar_view<int64_t, int64_t>> max_k)
{
  detail::eps_nn<int64_t, float>(handle, index, adj_ia, adj_ja, vd, query, eps, max_k);
}

void knn_query(raft::resources const& handle,
               const cuvs::neighbors::ball_cover::index<int64_t, float>& index,
               raft::device_matrix_view<const float, int64_t, raft::row_major> query,
               raft::device_matrix_view<int64_t, int64_t, raft::row_major> inds,
               raft::device_matrix_view<float, int64_t, raft::row_major> dists,
               bool perform_post_filtering,
               float weight)
{
  detail::knn_query<int64_t, float>(
    handle, index, query, inds, dists, perform_post_filtering, weight);
}

}  // namespace cuvs::neighbors::ball_cover
