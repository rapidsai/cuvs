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

#pragma once

#include "ann_types.hpp"
#include <cuvs/neighbors/ann_types.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>

namespace cuvs::neighbors::brute_force {

/**
 * @brief Brute Force index.
 *
 * The index stores the dataset and norms for the dataset in device memory.
 *
 * @tparam T data element type
 */

template <typename T>
struct index : cuvs::neighbors::ann::index {
 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;
  index(void* raft_index);

  cuvs::distance::DistanceType metric() const noexcept;
  size_t size() const noexcept;
  size_t dim() const noexcept;
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept;
  raft::device_vector_view<const T, int64_t, raft::row_major> norms() const;
  bool has_norms() const noexcept;
  T metric_arg() const noexcept;

  // Get pointer to underlying RAFT index, not meant to be used outside of cuVS
  inline const void* get_raft_index() const noexcept { return raft_index_.get(); }

 private:
  std::unique_ptr<void*> raft_index_;
};

auto build(raft::resources const& res,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
           float metric_arg                    = 0) -> cuvs::neighbors::brute_force::index<float>;

void search(raft::resources const& res,
            const cuvs::neighbors::brute_force::index<float>& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);

}  // namespace cuvs::neighbors::brute_force
