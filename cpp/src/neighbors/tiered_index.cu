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

namespace cuvs::neighbors::ivf_pq {
auto typed_build(raft::resources const& res,
                 const cuvs::neighbors::ivf_pq::index_params& index_params,
                 raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> cuvs::neighbors::ivf_pq::typed_index<float, int64_t>
{
  return static_cast<typed_index<float, int64_t>&&>(ivf_pq::build(res, index_params, dataset));
}

void typed_search(raft::resources const& res,
                  const ivf_pq::search_params& search_params,
                  const ivf_pq::typed_index<float, int64_t>& index,
                  raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
                  raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                  raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                  const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  ivf_pq::search(res, search_params, index, queries, neighbors, distances, sample_filter);
}
}  // namespace cuvs::neighbors::ivf_pq

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

auto build(raft::resources const& res,
           const index_params<ivf_pq::index_params>& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset)
  -> tiered_index::index<ivf_pq::typed_index<float, int64_t>>
{
  auto state =
    detail::build<ivf_pq::typed_index<float, int64_t>>(res, params, ivf_pq::typed_build, dataset);
  return cuvs::neighbors::tiered_index::index<ivf_pq::typed_index<float, int64_t>>(state);
}

void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<cagra::index<float, uint32_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(res, *idx->state, new_vectors);

  auto storage = next_state->storage;
  if (storage->num_rows_allocated != idx->state->storage->num_rows_allocated) {
    // CAGRA could be holding on to a non-owning view of the previous dataset in the ann_index,
    // which is problematic since the underlying ownership of the dataset could be freed here
    // call cagra::index::update_dataset on it to update the ann_index to point to the
    // new dataset
    if (next_state->ann_index) {
      auto dataset = raft::make_device_matrix_view<const float, int64_t>(
        storage->dataset.data(), next_state->ann_rows(), storage->dim);

      // Block 'search' calls during the update_dataset call to ensure that this
      // doesn't cause issues in a multithreaded environment
      std::unique_lock<std::shared_mutex> lock(idx->ann_mutex);
      next_state->ann_index->update_dataset(res, dataset);
    }
  }

  idx->state = next_state;
}

void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<ivf_flat::index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(res, *idx->state, new_vectors);
  idx->state      = next_state;
}

void extend(raft::resources const& res,
            raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
            tiered_index::index<ivf_pq::typed_index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::extend(res, *idx->state, new_vectors);
  idx->state      = next_state;
}

void compact(raft::resources const& res, tiered_index::index<cagra::index<float, uint32_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(res, *idx->state);
  idx->state      = next_state;
}

void compact(raft::resources const& res, tiered_index::index<ivf_flat::index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(res, *idx->state);
  idx->state      = next_state;
}

void compact(raft::resources const& res,
             tiered_index::index<ivf_pq::typed_index<float, int64_t>>* idx)
{
  std::scoped_lock lock(idx->write_mutex);
  auto next_state = detail::compact(res, *idx->state);
  idx->state      = next_state;
}

void search(raft::resources const& res,
            const cagra::search_params& search_params,
            const tiered_index::index<cagra::index<float, uint32_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  // use a read-write lock to handle calls to update_dataset for cagra
  // This allows multiple readers concurrently, but only one writer
  std::shared_lock<std::shared_mutex> lock(index.ann_mutex);
  index.state->search(
    res, search_params, cagra::search, queries, neighbors, distances, sample_filter);
}

void search(raft::resources const& res,
            const ivf_flat::search_params& search_params,
            const tiered_index::index<ivf_flat::index<float, int64_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  index.state->search(
    res, search_params, ivf_flat::search, queries, neighbors, distances, sample_filter);
}

void search(raft::resources const& res,
            const ivf_pq::search_params& search_params,
            const tiered_index::index<ivf_pq::typed_index<float, int64_t>>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter)
{
  index.state->search(
    res, search_params, ivf_pq::typed_search, queries, neighbors, distances, sample_filter);
}
}  // namespace cuvs::neighbors::tiered_index
