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

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/index_base.hpp>
#include <raft/core/device_mdspan.hpp>

#include <memory>
#include <vector>

namespace cuvs::neighbors::composite {

/**
 * @brief Composite index made of other IndexBase implementations.
 */
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
class CompositeIndex : public IndexBase<T, IdxT, OutputIdxT> {
 public:
  using value_type        = typename IndexBase<T, IdxT, OutputIdxT>::value_type;
  using index_type        = typename IndexBase<T, IdxT, OutputIdxT>::index_type;
  using out_index_type    = typename IndexBase<T, IdxT, OutputIdxT>::out_index_type;
  using matrix_index_type = typename IndexBase<T, IdxT, OutputIdxT>::matrix_index_type;

  using index_ptr = std::shared_ptr<IndexBase<value_type, index_type, out_index_type>>;

  explicit CompositeIndex(std::vector<index_ptr> children) : children_(std::move(children)) {}

  void search(
    const raft::resources& handle,
    const cuvs::neighbors::search_params& params,
    raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> queries,
    raft::device_matrix_view<out_index_type, matrix_index_type, raft::row_major> neighbors,
    raft::device_matrix_view<float, matrix_index_type, raft::row_major> distances,
    const cuvs::neighbors::filtering::base_filter& filter =
      cuvs::neighbors::filtering::none_sample_filter{}) const override;

  index_type size() const noexcept override
  {
    index_type total = 0;
    for (const auto& c : children_) {
      total += c->size();
    }
    return total;
  }

  cuvs::distance::DistanceType metric() const noexcept override
  {
    return children_.empty() ? cuvs::distance::DistanceType::L2Expanded
                             : children_.front()->metric();
  }

 private:
  std::vector<index_ptr> children_;
};

}  // namespace cuvs::neighbors::composite
