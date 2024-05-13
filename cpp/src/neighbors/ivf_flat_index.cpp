/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/ivf_flat.hpp>

namespace cuvs::neighbors::ivf_flat {

template <typename T, typename IdxT>
index<T, IdxT>::index(raft::resources const& res, const index_params& params, uint32_t dim)
  : ann::index(),
    raft_index_(std::make_unique<raft::neighbors::ivf_flat::index<T, IdxT>>(
      res,
      static_cast<raft::distance::DistanceType>((int)params.metric),
      params.n_lists,
      params.adaptive_centers,
      params.conservative_memory_allocation,
      dim))
{
}

template <typename T, typename IdxT>
index<T, IdxT>::index(raft::neighbors::ivf_flat::index<T, IdxT>&& raft_idx)
  : ann::index(),
    raft_index_(std::make_unique<raft::neighbors::ivf_flat::index<T, IdxT>>(std::move(raft_idx)))
{
}

template <typename T, typename IdxT>
uint32_t index<T, IdxT>::veclen() const noexcept
{
  return raft_index_->veclen();
}

template <typename T, typename IdxT>
cuvs::distance::DistanceType index<T, IdxT>::metric() const noexcept
{
  return static_cast<cuvs::distance::DistanceType>((int)raft_index_->metric());
}

template <typename T, typename IdxT>
bool index<T, IdxT>::adaptive_centers() const noexcept
{
  return raft_index_->adaptive_centers();
}

template <typename T, typename IdxT>
raft::device_vector_view<uint32_t, uint32_t> index<T, IdxT>::list_sizes() noexcept
{
  return raft_index_->list_sizes();
}

template <typename T, typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t> index<T, IdxT>::list_sizes() const noexcept
{
  return raft_index_->list_sizes();
}

template <typename T, typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<T, IdxT>::centers() noexcept
{
  return raft_index_->centers();
}

template <typename T, typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<T, IdxT>::centers()
  const noexcept
{
  return raft_index_->centers();
}

template <typename T, typename IdxT>
std::optional<raft::device_vector_view<float, uint32_t>> index<T, IdxT>::center_norms() noexcept
{
  return raft_index_->center_norms();
}

template <typename T, typename IdxT>
std::optional<raft::device_vector_view<const float, uint32_t>> index<T, IdxT>::center_norms()
  const noexcept
{
  return raft_index_->center_norms();
}

template <typename T, typename IdxT>
IdxT index<T, IdxT>::size() const noexcept
{
  return raft_index_->size();
}

template <typename T, typename IdxT>
uint32_t index<T, IdxT>::dim() const noexcept
{
  return raft_index_->dim();
}

template <typename T, typename IdxT>
uint32_t index<T, IdxT>::n_lists() const noexcept
{
  return raft_index_->n_lists();
}

template <typename T, typename IdxT>
raft::device_vector_view<T*, uint32_t> index<T, IdxT>::data_ptrs() noexcept
{
  return raft_index_->data_ptrs();
}

template <typename T, typename IdxT>
raft::device_vector_view<T* const, uint32_t> index<T, IdxT>::data_ptrs() const noexcept
{
  return raft_index_->data_ptrs();
}

template <typename T, typename IdxT>
raft::device_vector_view<IdxT*, uint32_t> index<T, IdxT>::inds_ptrs() noexcept
{
  return raft_index_->inds_ptrs();
}

template <typename T, typename IdxT>
raft::device_vector_view<IdxT* const, uint32_t> index<T, IdxT>::inds_ptrs() const noexcept
{
  return raft_index_->inds_ptrs();
}

template <typename T, typename IdxT>
bool index<T, IdxT>::conservative_memory_allocation() const noexcept
{
  return raft_index_->conservative_memory_allocation();
}

template <typename T, typename IdxT>
std::vector<std::shared_ptr<raft::neighbors::ivf_flat::list_data<T, IdxT>>>&
index<T, IdxT>::lists() noexcept
{
  return raft_index_->lists();
}

template <typename T, typename IdxT>
const std::vector<std::shared_ptr<raft::neighbors::ivf_flat::list_data<T, IdxT>>>&
index<T, IdxT>::lists() const noexcept
{
  return raft_index_->lists();
}

template struct index<float, int64_t>;
template struct index<int8_t, int64_t>;
template struct index<uint8_t, int64_t>;

}  // namespace cuvs::neighbors::ivf_flat
