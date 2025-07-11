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

#include <cuvs/neighbors/binary_ivf.hpp>

namespace cuvs::neighbors::binary_ivf {

template <typename IdxT>
index<IdxT>::index(raft::resources const& res)
  : index(res, cuvs::distance::DistanceType::L2Expanded, 0, false, false, 0)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& res, const index_params& params, uint32_t dim)
  : index(res,
          params.metric,
          params.n_lists,
          params.adaptive_centers,
          params.conservative_memory_allocation,
          dim)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& res,
                      uint32_t n_lists,
                      bool adaptive_centers,
                      bool conservative_memory_allocation,
                      uint32_t dim)
  : cuvs::neighbors::index(),
    adaptive_centers_(adaptive_centers),
    conservative_memory_allocation_{conservative_memory_allocation},
    lists_{n_lists},
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(res, n_lists)},
    centers_(raft::make_device_matrix<uint8_t, uint32_t>(res, n_lists, dim)),
    center_norms_(std::nullopt),
    data_ptrs_{raft::make_device_vector<uint8_t*, uint32_t>(res, n_lists)},
    inds_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(res, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<IdxT, uint32_t>(n_lists + 1)}
{
  check_consistency();
  accum_sorted_sizes_(n_lists) = 0;
}

template <typename IdxT>
bool index<IdxT>::adaptive_centers() const noexcept
{
  return adaptive_centers_;
}

template <typename IdxT>
raft::device_vector_view<uint32_t, uint32_t> index<IdxT>::list_sizes() noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<const uint32_t, uint32_t> index<IdxT>::list_sizes() const noexcept
{
  return list_sizes_.view();
}

template <typename IdxT>
raft::device_matrix_view<uint8_t, uint32_t, raft::row_major> index<IdxT>::centers() noexcept
{
  return centers_.view();
}

template <typename IdxT>
raft::device_matrix_view<const uint8_t, uint32_t, raft::row_major> index<IdxT>::centers()
  const noexcept
{
  return centers_.view();
}

template <typename IdxT>
auto index<IdxT>::accum_sorted_sizes() noexcept -> raft::host_vector_view<IdxT, uint32_t>
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
[[nodiscard]] auto index<IdxT>::accum_sorted_sizes() const noexcept
  -> raft::host_vector_view<const IdxT, uint32_t>
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
IdxT index<IdxT>::size() const noexcept
{
  return accum_sorted_sizes()(n_lists());
}

template <typename IdxT>
uint32_t index<IdxT>::dim() const noexcept
{
  return centers_.extent(1);
}

template <typename IdxT>
uint32_t index<IdxT>::n_lists() const noexcept
{
  return lists_.size();
}

template <typename IdxT>
raft::device_vector_view<uint8_t*, uint32_t> index<IdxT>::data_ptrs() noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<uint8_t* const, uint32_t> index<IdxT>::data_ptrs() const noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<IdxT*, uint32_t> index<IdxT>::inds_ptrs() noexcept
{
  return inds_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<IdxT* const, uint32_t> index<IdxT>::inds_ptrs() const noexcept
{
  return inds_ptrs_.view();
}

template <typename IdxT>
bool index<IdxT>::conservative_memory_allocation() const noexcept
{
  return conservative_memory_allocation_;
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
void index<IdxT>::check_consistency()
{
  auto n_lists = lists_.size();
//   RAFT_EXPECTS(dim() % veclen_ == 0, "dimensionality is not a multiple of the veclen");
  RAFT_EXPECTS(list_sizes_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(data_ptrs_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(inds_ptrs_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(                                       //
    (centers_.extent(0) == list_sizes_.extent(0)) &&  //
      (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
    "inconsistent number of lists (clusters)");
}

template struct index<uint32_t>;  // Used for refine function
template struct index<int64_t>;

}  // namespace cuvs::neighbors::ivf_flat
