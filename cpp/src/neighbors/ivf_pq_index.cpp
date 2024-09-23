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
index_params index_params::from_dataset(raft::matrix_extent<int64_t> dataset,
                                        cuvs::distance::DistanceType metric)
{
  index_params params;
  params.n_lists =
    dataset.extent(0) < 4 * 2500 ? 4 : static_cast<uint32_t>(std::sqrt(dataset.extent(0)));
  params.n_lists = std::min<uint32_t>(params.n_lists, dataset.extent(0));
  params.pq_dim =
    raft::round_up_safe(static_cast<uint32_t>(dataset.extent(1) / 4), static_cast<uint32_t>(8));
  if (params.pq_dim == 0) params.pq_dim = 8;
  params.pq_bits                  = 8;
  params.kmeans_trainset_fraction = dataset.extent(0) < 10000 ? 1 : 0.1;
  params.metric                   = metric;
  return params;
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle)
  // this constructor is just for a temporary index, for use in the deserialization
  // api. all the parameters here will get replaced with loaded values - that aren't
  // necessarily known ahead of time before deserialization.
  // TODO: do we even need a handle here - could just construct one?
  : index(handle,
          cuvs::distance::DistanceType::L2Expanded,
          codebook_gen::PER_SUBSPACE,
          0,
          0,
          8,
          0,
          true)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle, const index_params& params, uint32_t dim)
  : index(handle,
          params.metric,
          params.codebook_kind,
          params.n_lists,
          dim,
          params.pq_bits,
          params.pq_dim,
          params.conservative_memory_allocation)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& handle,
                   cuvs::distance::DistanceType metric,
                   codebook_gen codebook_kind,
                   uint32_t n_lists,
                   uint32_t dim,
                   uint32_t pq_bits,
                   uint32_t pq_dim,
                   bool conservative_memory_allocation)
  : cuvs::neighbors::index(),
    metric_(metric),
    codebook_kind_(codebook_kind),
    dim_(dim),
    pq_bits_(pq_bits),
    pq_dim_(pq_dim == 0 ? calculate_pq_dim(dim) : pq_dim),
    conservative_memory_allocation_(conservative_memory_allocation),
    lists_{n_lists},
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(handle, n_lists)},
    pq_centers_{raft::make_device_mdarray<float>(handle, make_pq_centers_extents())},
    centers_{raft::make_device_matrix<float, uint32_t>(handle, n_lists, this->dim_ext())},
    centers_rot_{raft::make_device_matrix<float, uint32_t>(handle, n_lists, this->rot_dim())},
    rotation_matrix_{
      raft::make_device_matrix<float, uint32_t>(handle, this->rot_dim(), this->dim())},
    data_ptrs_{raft::make_device_vector<uint8_t*, uint32_t>(handle, n_lists)},
    inds_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(handle, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<IdxT, uint32_t>(n_lists + 1)}
{
  check_consistency();
  accum_sorted_sizes_(n_lists) = 0;
}

template <typename IdxT>
IdxT index<IdxT>::size() const noexcept
{
  return accum_sorted_sizes_(n_lists());
}

template <typename IdxT>
uint32_t index<IdxT>::dim() const noexcept
{
  return dim_;
}

template <typename IdxT>
uint32_t index<IdxT>::dim_ext() const noexcept
{
  return raft::round_up_safe(dim() + 1, 8u);
}

template <typename IdxT>
uint32_t index<IdxT>::rot_dim() const noexcept
{
  return pq_len() * pq_dim();
}

template <typename IdxT>
uint32_t index<IdxT>::pq_bits() const noexcept
{
  return pq_bits_;
}

template <typename IdxT>
uint32_t index<IdxT>::pq_dim() const noexcept
{
  return pq_dim_;
}

template <typename IdxT>
uint32_t index<IdxT>::pq_len() const noexcept
{
  return raft::div_rounding_up_unsafe(dim(), pq_dim());
}

template <typename IdxT>
uint32_t index<IdxT>::pq_book_size() const noexcept
{
  return 1 << pq_bits();
}

template <typename IdxT>
cuvs::distance::DistanceType index<IdxT>::metric() const noexcept
{
  return metric_;
}

template <typename IdxT>
codebook_gen index<IdxT>::codebook_kind() const noexcept
{
  return codebook_kind_;
}

template <typename IdxT>
uint32_t index<IdxT>::n_lists() const noexcept
{
  return lists_.size();
}

template <typename IdxT>
bool index<IdxT>::conservative_memory_allocation() const noexcept
{
  return conservative_memory_allocation_;
}

template <typename IdxT>
raft::device_mdspan<float,
                    typename cuvs::neighbors::ivf_pq::index<IdxT>::pq_centers_extents,
                    raft::row_major>
index<IdxT>::pq_centers() noexcept
{
  return pq_centers_.view();
}

template <typename IdxT>
raft::device_mdspan<const float,
                    typename cuvs::neighbors::ivf_pq::index<IdxT>::pq_centers_extents,
                    raft::row_major>
index<IdxT>::pq_centers() const noexcept
{
  return pq_centers_.view();
}

template <typename IdxT>
std::vector<std::shared_ptr<list_data<IdxT>>>& index<IdxT>::lists() noexcept
{
  return lists_;
}

template <typename IdxT>
const std::vector<std::shared_ptr<list_data<IdxT>>>& index<IdxT>::lists() const noexcept
{
  return lists_;
}

template <typename IdxT>
raft::device_vector_view<uint8_t*, uint32_t, raft::row_major> index<IdxT>::data_ptrs() noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<const uint8_t* const, uint32_t, raft::row_major> index<IdxT>::data_ptrs()
  const noexcept
{
  return raft::make_mdspan<const uint8_t* const, uint32_t, raft::row_major, false, true>(
    data_ptrs_.data_handle(), data_ptrs_.extents());
}

template <typename IdxT>
raft::device_vector_view<IdxT*, uint32_t, raft::row_major> index<IdxT>::inds_ptrs() noexcept
{
  return inds_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<const IdxT* const, uint32_t, raft::row_major> index<IdxT>::inds_ptrs()
  const noexcept
{
  return raft::make_mdspan<const IdxT* const, uint32_t, raft::row_major, false, true>(
    inds_ptrs_.data_handle(), inds_ptrs_.extents());
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::rotation_matrix() noexcept
{
  return rotation_matrix_.view();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::rotation_matrix()
  const noexcept
{
  return rotation_matrix_.view();
}

template <typename IdxT>
raft::host_vector_view<IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes() noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
raft::host_vector_view<const IdxT, uint32_t, raft::row_major> index<IdxT>::accum_sorted_sizes()
  const noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes() noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t, raft::row_major> index<IdxT>::list_sizes()
  const noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::centers() noexcept
{
  return centers_.view();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers()
  const noexcept
{
  return centers_.view();
}

template <typename IdxT>
raft::device_matrix_view<float, uint32_t, raft::row_major> index<IdxT>::centers_rot() noexcept
{
  return centers_rot_.view();
}

template <typename IdxT>
raft::device_matrix_view<const float, uint32_t, raft::row_major> index<IdxT>::centers_rot()
  const noexcept
{
  return centers_rot_.view();
}

template <typename IdxT>
uint32_t index<IdxT>::get_list_size_in_bytes(uint32_t label)
{
  RAFT_EXPECTS(label < this->n_lists(),
               "Expected label to be less than number of lists in the index");
  auto list_data = this->lists()[label]->data;
  return list_data.size();
}

template <typename IdxT>
void index<IdxT>::check_consistency()
{
  RAFT_EXPECTS(pq_bits() >= 4 && pq_bits() <= 8,
               "`pq_bits` must be within closed range [4,8], but got %u.",
               pq_bits());
  RAFT_EXPECTS((pq_bits() * pq_dim()) % 8 == 0,
               "`pq_bits * pq_dim` must be a multiple of 8, but got %u * %u = %u.",
               pq_bits(),
               pq_dim(),
               pq_bits() * pq_dim());
}

template <typename IdxT>
typename index<IdxT>::pq_centers_extents index<IdxT>::make_pq_centers_extents()
{
  switch (codebook_kind()) {
    case codebook_gen::PER_SUBSPACE:
      return raft::make_extents<uint32_t>(pq_dim(), pq_len(), pq_book_size());
    case codebook_gen::PER_CLUSTER:
      return raft::make_extents<uint32_t>(n_lists(), pq_len(), pq_book_size());
    default: RAFT_FAIL("Unreachable code");
  }
}

template <typename IdxT>
uint32_t index<IdxT>::calculate_pq_dim(uint32_t dim)
{
  // If the dimensionality is large enough, we can reduce it to improve performance
  if (dim >= 128) { dim /= 2; }
  // Round it down to 32 to improve performance.
  auto r = raft::round_down_safe<uint32_t>(dim, 32);
  if (r > 0) return r;
  // If the dimensionality is really low, round it to the closest power-of-two
  r = 1;
  while ((r << 1) <= dim) {
    r = r << 1;
  }
  return r;
}

template struct index<int64_t>;

}  // namespace cuvs::neighbors::ivf_pq
