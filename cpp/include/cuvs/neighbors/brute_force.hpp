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
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;
  cuvs::distance::DistanceType metric() const noexcept;
  size_t size() const noexcept;
  size_t dim() const noexcept;
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept;
  raft::device_vector_view<const T, int64_t, raft::row_major> norms() const;
  bool has_norms() const noexcept;
  T metric_arg() const noexcept;
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset);

 private:
  std::unique_ptr<void*> raft_index_;
};

#define CUVS_INST_BFKNN(T, IdxT)                                                               \
  auto build(raft::resources const& res,                                                       \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,                 \
             cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded, \
             T metric_arg                        = 0)                                          \
    ->cuvs::neighbors::brute_force::index<T>;                                                  \
                                                                                               \
  void search(raft::resources const& res,                                                      \
              const cuvs::neighbors::brute_force::index<T>& idx,                               \
              raft::device_matrix_view<const T, IdxT, raft::row_major> queries,                \
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,                 \
              raft::device_matrix_view<T, IdxT, raft::row_major> distances);

CUVS_INST_BFKNN(float, int64_t);
// CUVS_INST_BFKNN(int8_t, int64_t);
// CUVS_INST_BFKNN(uint8_t, int64_t);

}  // namespace cuvs::neighbors::brute_force