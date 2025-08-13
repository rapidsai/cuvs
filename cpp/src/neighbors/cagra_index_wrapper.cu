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

#include <cuda_fp16.h>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/cagra_index_wrapper.hpp>
#include <cuvs/neighbors/composite/index.hpp>

namespace cuvs::neighbors::cagra {

template <typename T, typename IdxT, typename OutputIdxT>
IndexWrapper<T, IdxT, OutputIdxT>::IndexWrapper(cuvs::neighbors::cagra::index<T, IdxT>* idx)
  : index_(idx)
{
}

template <typename T, typename IdxT, typename OutputIdxT>
void IndexWrapper<T, IdxT, OutputIdxT>::search(
  const raft::resources& handle,
  const cuvs::neighbors::search_params& params,
  raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> queries,
  raft::device_matrix_view<out_index_type, matrix_index_type, raft::row_major> neighbors,
  raft::device_matrix_view<float, matrix_index_type, raft::row_major> distances,
  const cuvs::neighbors::filtering::base_filter& filter) const
{
  auto const& cagra_params = static_cast<const cuvs::neighbors::cagra::search_params&>(params);
  cuvs::neighbors::cagra::search(
    handle, cagra_params, *index_, queries, neighbors, distances, filter);
}

template <typename T, typename IdxT, typename OutputIdxT>
typename IndexWrapper<T, IdxT, OutputIdxT>::index_type IndexWrapper<T, IdxT, OutputIdxT>::size()
  const noexcept
{
  return index_->size();
}

template <typename T, typename IdxT, typename OutputIdxT>
cuvs::distance::DistanceType IndexWrapper<T, IdxT, OutputIdxT>::metric() const noexcept
{
  return index_->metric();
}

template <typename T, typename IdxT, typename OutputIdxT>
std::shared_ptr<
  cuvs::neighbors::IndexBase<typename IndexWrapper<T, IdxT, OutputIdxT>::value_type,
                             typename IndexWrapper<T, IdxT, OutputIdxT>::index_type,
                             typename IndexWrapper<T, IdxT, OutputIdxT>::out_index_type>>
IndexWrapper<T, IdxT, OutputIdxT>::merge(
  const raft::resources& handle,
  const cuvs::neighbors::merge_params& params,
  const std::vector<
    std::shared_ptr<cuvs::neighbors::IndexBase<value_type, index_type, out_index_type>>>&
    other_indices) const
{
  const auto* cagra_params = dynamic_cast<const cuvs::neighbors::cagra::merge_params*>(&params);
  if (!cagra_params) { RAFT_FAIL("CAGRA IndexWrapper::merge requires cagra::merge_params"); }

  std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> cagra_indices;
  cagra_indices.push_back(index_);

  for (const auto& other : other_indices) {
    const auto* other_wrapper = dynamic_cast<const IndexWrapper<T, IdxT, OutputIdxT>*>(other.get());
    if (!other_wrapper) {
      RAFT_FAIL("CAGRA IndexWrapper::merge can only merge with other CAGRA indices");
    }
    cagra_indices.push_back(other_wrapper->index_);
  }

  if (cagra_params->strategy() == cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL) {
    std::vector<std::shared_ptr<cuvs::neighbors::IndexBase<T, IdxT, OutputIdxT>>> wrappers;
    wrappers.reserve(cagra_indices.size());
    for (auto* idx : cagra_indices) {
      wrappers.push_back(std::make_shared<IndexWrapper<T, IdxT, OutputIdxT>>(idx));
    }
    return std::make_shared<cuvs::neighbors::composite::CompositeIndex<T, IdxT, OutputIdxT>>(
      std::move(wrappers));
  } else if (cagra_params->strategy() == cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_PHYSICAL) {
    auto merged_index = cuvs::neighbors::cagra::merge(handle, *cagra_params, cagra_indices);
    auto* idx         = new decltype(merged_index)(std::move(merged_index));
    return std::make_shared<IndexWrapper<T, IdxT, OutputIdxT>>(idx);
  }

  RAFT_FAIL("Invalid merge strategy");
}

template class IndexWrapper<float, uint32_t, uint32_t>;
template class IndexWrapper<half, uint32_t, uint32_t>;
template class IndexWrapper<int8_t, uint32_t, uint32_t>;
template class IndexWrapper<uint8_t, uint32_t, uint32_t>;

template class IndexWrapper<float, uint32_t, int64_t>;
template class IndexWrapper<half, uint32_t, int64_t>;
template class IndexWrapper<int8_t, uint32_t, int64_t>;
template class IndexWrapper<uint8_t, uint32_t, int64_t>;

}  // namespace cuvs::neighbors::cagra
