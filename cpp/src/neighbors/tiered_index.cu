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

#include "detail/tiered_index.cuh"

#include <cuvs/neighbors/tiered_index.hpp>

namespace cuvs::neighbors::tiered_index {
auto build(raft::resources const& res,
           const index_params<cagra::index_params>& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<cagra::index<float, uint32_t>>
{
  auto state = detail::build<cagra::index<float, uint32_t>>(res, params, cagra::build, dataset);
  return cuvs::neighbors::tiered_index::index<cagra::index<float, uint32_t>>(state);
}

auto build(raft::resources const& res,
           const index_params<ivf_flat::index_params>& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<ivf_flat::index<float, int64_t>>
{
  auto state =
    detail::build<ivf_flat::index<float, int64_t>>(res, params, ivf_flat::build, dataset);
  return cuvs::neighbors::tiered_index::index<ivf_flat::index<float, int64_t>>(state);
}

void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<cagra::index<float, uint32_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(handle, *idx->state, new_vectors);
  idx->state      = next_state;
}

void extend(raft::resources const& handle,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<ivf_flat::index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(handle, *idx->state, new_vectors);
  idx->state      = next_state;
}

void compact(raft::resources const& handle, tiered_index::index<cagra::index<float, uint32_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(handle, *idx->state);
  idx->state      = next_state;
}

void compact(raft::resources const& handle,
             tiered_index::index<ivf_flat::index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(handle, *idx->state);
  idx->state      = next_state;
}

void search(raft::resources const& handle,
            const cagra::search_params& search_params,
            const tiered_index::index<cagra::index<float, uint32_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  index.state->search(
    handle, search_params, cagra::search, queries, neighbors, distances, sample_filter);
}

void search(raft::resources const& handle,
            const ivf_flat::search_params& search_params,
            const tiered_index::index<ivf_flat::index<float, int64_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  index.state->search(
    handle, search_params, ivf_flat::search, queries, neighbors, distances, sample_filter);
}
}  // namespace cuvs::neighbors::tiered_index
