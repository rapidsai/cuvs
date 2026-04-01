/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/ivf_sq.hpp>

namespace cuvs::neighbors::ivf_sq {

template <typename IdxT>
index<IdxT>::index(raft::resources const& res)
  : index(res, cuvs::distance::DistanceType::L2Expanded, 0, 0, false, false)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& res, const index_params& params, uint32_t dim)
  : index(res,
          params.metric,
          params.n_lists,
          dim,
          params.adaptive_centers,
          params.conservative_memory_allocation)
{
}

template <typename IdxT>
index<IdxT>::index(raft::resources const& res,
                   cuvs::distance::DistanceType metric,
                   uint32_t n_lists,
                   uint32_t dim,
                   bool adaptive_centers,
                   bool conservative_memory_allocation)
  : cuvs::neighbors::index(),
    metric_(metric),
    adaptive_centers_(adaptive_centers),
    conservative_memory_allocation_(conservative_memory_allocation),
    lists_{n_lists},
    list_sizes_{raft::make_device_vector<uint32_t, uint32_t>(res, n_lists)},
    centers_(raft::make_device_matrix<float, uint32_t>(res, n_lists, dim)),
    center_norms_(std::nullopt),
    sq_vmin_{raft::make_device_vector<float, uint32_t>(res, dim)},
    sq_delta_{raft::make_device_vector<float, uint32_t>(res, dim)},
    data_ptrs_{raft::make_device_vector<IdxT*, uint32_t>(res, n_lists)},
    inds_ptrs_{raft::make_device_vector<int64_t*, uint32_t>(res, n_lists)},
    accum_sorted_sizes_{raft::make_host_vector<int64_t, uint32_t>(n_lists + 1)}
{
  check_consistency();
  accum_sorted_sizes_(n_lists) = 0;
}

template <typename IdxT>
cuvs::distance::DistanceType index<IdxT>::metric() const noexcept
{
  return metric_;
}

template <typename IdxT>
bool index<IdxT>::adaptive_centers() const noexcept
{
  return adaptive_centers_;
}

template <typename IdxT>
int64_t index<IdxT>::size() const noexcept
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
bool index<IdxT>::conservative_memory_allocation() const noexcept
{
  return conservative_memory_allocation_;
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
std::optional<raft::device_vector_view<float, uint32_t>> index<IdxT>::center_norms() noexcept
{
  if (center_norms_.has_value()) {
    return std::make_optional<raft::device_vector_view<float, uint32_t>>(center_norms_->view());
  } else {
    return std::nullopt;
  }
}

template <typename IdxT>
std::optional<raft::device_vector_view<const float, uint32_t>> index<IdxT>::center_norms()
  const noexcept
{
  if (center_norms_.has_value()) {
    return std::make_optional<raft::device_vector_view<const float, uint32_t>>(
      center_norms_->view());
  } else {
    return std::nullopt;
  }
}

template <typename IdxT>
void index<IdxT>::allocate_center_norms(raft::resources const& res)
{
  switch (metric_) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2Unexpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
    case cuvs::distance::DistanceType::CosineExpanded:
      center_norms_ = raft::make_device_vector<float, uint32_t>(res, n_lists());
      break;
    default: center_norms_ = std::nullopt;
  }
}

template <typename IdxT>
raft::device_vector_view<float, uint32_t> index<IdxT>::sq_vmin() noexcept
{
  return sq_vmin_.view();
}

template <typename IdxT>
raft::device_vector_view<const float, uint32_t> index<IdxT>::sq_vmin() const noexcept
{
  return sq_vmin_.view();
}

template <typename IdxT>
raft::device_vector_view<float, uint32_t> index<IdxT>::sq_delta() noexcept
{
  return sq_delta_.view();
}

template <typename IdxT>
raft::device_vector_view<const float, uint32_t> index<IdxT>::sq_delta() const noexcept
{
  return sq_delta_.view();
}

template <typename IdxT>
raft::host_vector_view<int64_t, uint32_t> index<IdxT>::accum_sorted_sizes() noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
raft::host_vector_view<const int64_t, uint32_t> index<IdxT>::accum_sorted_sizes() const noexcept
{
  return accum_sorted_sizes_.view();
}

template <typename IdxT>
raft::device_vector_view<IdxT*, uint32_t> index<IdxT>::data_ptrs() noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<IdxT* const, uint32_t> index<IdxT>::data_ptrs() const noexcept
{
  return data_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<int64_t*, uint32_t> index<IdxT>::inds_ptrs() noexcept
{
  return inds_ptrs_.view();
}

template <typename IdxT>
raft::device_vector_view<int64_t* const, uint32_t> index<IdxT>::inds_ptrs() const noexcept
{
  return inds_ptrs_.view();
}

template <typename IdxT>
std::vector<std::shared_ptr<list_data<IdxT, int64_t>>>& index<IdxT>::lists() noexcept
{
  return lists_;
}

template <typename IdxT>
const std::vector<std::shared_ptr<list_data<IdxT, int64_t>>>& index<IdxT>::lists() const noexcept
{
  return lists_;
}

template <typename IdxT>
void index<IdxT>::check_consistency()
{
  auto n_lists = lists_.size();
  RAFT_EXPECTS(list_sizes_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(data_ptrs_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS(inds_ptrs_.extent(0) == n_lists, "inconsistent list size");
  RAFT_EXPECTS((centers_.extent(0) == list_sizes_.extent(0)) &&
                 (!center_norms_.has_value() || centers_.extent(0) == center_norms_->extent(0)),
               "inconsistent number of lists (clusters)");
}

template struct index<uint8_t>;

}  // namespace cuvs::neighbors::ivf_sq
