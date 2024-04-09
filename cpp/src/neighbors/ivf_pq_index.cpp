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

#include <cuvs/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq {

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle, const index_params& params, uint32_t dim)
  : ann::index(),
    raft_index_(std::make_unique<raft::neighbors::ivf_pq::index<IdxT>>(
      handle,
      static_cast<raft::distance::DistanceType>((int)params.metric),
      static_cast<raft::neighbors::ivf_pq::codebook_gen>((int)params.codebook_kind),
      params.n_lists,
      dim,
      params.pq_bits,
      params.pq_dim,
      params.conservative_memory_allocation))
{
}

template <typename IdxT>
index<IdxT>::index(raft::neighbors::ivf_pq::index<IdxT>&& raft_idx)
  : ann::index(),
    raft_index_(std::make_unique<raft::neighbors::ivf_pq::index<IdxT>>(std::move(raft_idx)))
{
}

template <typename IdxT>
IdxT index<IdxT>::size() const noexcept
{
  return raft_index_->size();
}

template <typename IdxT>
uint32_t index<IdxT>::dim() const noexcept
{
  return raft_index_->dim();
}

template <typename IdxT>
uint32_t index<IdxT>::dim_ext() const noexcept
{
  return raft_index_->dim_ext();
}

template <typename IdxT>
uint32_t index<IdxT>::rot_dim() const noexcept
{
  return raft_index_->rot_dim();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_bits() const noexcept
{
  return raft_index_->pq_bits();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_dim() const noexcept
{
  return raft_index_->pq_dim();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_len() const noexcept
{
  return raft_index_->pq_len();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_book_size() const noexcept
{
  return raft_index_->pq_book_size();
}

template <typename IdxT>
cuvs::distance::DistanceType index<IdxT>::metric() const noexcept
{
  return static_cast<cuvs::distance::DistanceType>((int)raft_index_->metric());
}

template <typename IdxT>
codebook_gen index<IdxT>::codebook_kind() const noexcept
{
  return static_cast<codebook_gen>((int)raft_index_->codebook_kind());
}

template <typename IdxT>
uint32_t index<IdxT>::n_lists() const noexcept
{
  return raft_index_->n_lists();
}

template <typename IdxT>
bool index<IdxT>::conservative_memory_allocation() const noexcept
{
  return raft_index_->conservative_memory_allocation();
}

template <typename IdxT>
raft::
  mdspan<float, typename cuvs::neighbors::ivf_pq::index<IdxT>::pq_centers_extents, raft::row_major>
  index<IdxT>::pq_centers() noexcept
{
  return raft_index_->pq_centers();
}

template <typename IdxT>
raft::mdspan<const float,
             typename cuvs::neighbors::ivf_pq::index<IdxT>::pq_centers_extents,
             raft::row_major>
index<IdxT>::pq_centers() const noexcept
{
  return raft_index_->pq_centers();
}

template <typename IdxT>
std::vector<std::shared_ptr<list_data<IdxT>>>& index<IdxT>::lists() noexcept
{
  return raft_index_->lists();
}

template <typename IdxT>
const std::vector<std::shared_ptr<list_data<IdxT>>>& index<IdxT>::lists() const noexcept
{
  return raft_index_->lists();
}

template <typename IdxT>
raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> index<IdxT>::data_ptrs() noexcept
{
  return raft_index_->data_ptrs();
}

template <typename IdxT>
raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> index<IdxT>::data_ptrs()
  const noexcept
{
  return raft_index_->data_ptrs();
}

template <typename IdxT>
raft::device_vector_view<IdxT*, uint32_t, raft::row_major> index<IdxT>::inds_ptrs() noexcept
{
  return raft_index_->inds_ptrs();
}

template <typename IdxT>
raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> index<IdxT>::inds_ptrs()
  const noexcept
{
  return raft_index_->inds_ptrs();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::rotation_matrix() noexcept
{
  return raft_index_->rotation_matrix();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::rotation_matrix()
  const noexcept
{
  return raft_index_->rotation_matrix();
}

template <typename IdxT>
raft::host_vector_view<IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes() noexcept
{
  return raft_index_->accum_sorted_sizes();
}

template <typename IdxT>
raft::host_vector_view<const IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes()
  const noexcept
{
  return raft_index_->accum_sorted_sizes();
}

template <typename IdxT>
raft::device_vector_view<uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes() noexcept
{
  return raft_index_->list_sizes();
}

template <typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes()
  const noexcept
{
  return raft_index_->list_sizes();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::centers() noexcept
{
  return raft_index_->centers();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers()
  const noexcept
{
  return raft_index_->centers();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::centers_rot() noexcept
{
  return raft_index_->centers_rot();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers_rot()
  const noexcept
{
  return raft_index_->centers_rot();
}

template struct index<int64_t>;

}  // namespace cuvs::neighbors::ivf_pq
