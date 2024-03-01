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
#include <raft/neighbors/brute_force-inl.cuh>

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
  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

  /** Build a cuvs bruteforce index from an existing RAFT bruteforce index. */
  index(raft::neighbors::brute_force::index<T>&& raft_idx)
    : cuvs::neighbors::ann::index(),
      raft_index_(std::make_unique<raft::neighbors::brute_force::index<T>>(std::move(raft_idx)))
  {
  }

  /** Distance metric used for retrieval */
  [[nodiscard]] constexpr inline cuvs::distance::DistanceType metric() const noexcept
  {
    return raft_index_->metric_;
  }

  /** Total length of the index (number of vectors). */
  [[nodiscard]] constexpr inline auto size() const noexcept
  {
    return raft_index_->dataset_view_.extent(0);
  }

  /** Dimensionality of the data. */
  [[nodiscard]] constexpr inline auto dim() const noexcept
  {
    return raft_index_->dataset_view_.extent(1);
  }

  /** Dataset [size, dim] */
  [[nodiscard]] inline auto dataset() const noexcept
    -> raft::device_matrix_view<const T, int64_t, raft::row_major>
  {
    return raft_index_->dataset_view_;
  }

  /** Dataset norms */
  [[nodiscard]] inline auto norms() const
    -> raft::device_vector_view<const T, int64_t, raft::row_major>
  {
    return raft_index_->norms_view_.value();
  }

  /** Whether or not this index has dataset norms */
  [[nodiscard]] inline bool has_norms() const noexcept
  {
    return raft_index_->norms_view_.has_value();
  }

  [[nodiscard]] inline T metric_arg() const noexcept { return raft_index_->metric_arg_; }

  /**
   * Replace the dataset with a new dataset.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)
  {
    raft_index_->dataset_view_ = dataset;
  }

  auto get_raft_index() const -> const raft::neighbors::brute_force::index<T>*
  {
    return raft_index_.get();
  }
  auto get_raft_index() -> raft::neighbors::brute_force::index<T>* { return raft_index_.get(); }

 private:
  std::unique_ptr<raft::neighbors::brute_force::index<T>> raft_index_;
};

#define CUVS_INST_BFKNN(T, IdxT)                                                               \
  auto build(raft::resources const& res,                                                       \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,                 \
             cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded, \
             float metric_arg                    = 0.0)                                        \
    ->cuvs::neighbors::brute_force::index<T>;                                                  \
                                                                                               \
  void search(raft::resources const& res,                                                      \
              const cuvs::neighbors::brute_force::index<T>& idx,                               \
              raft::device_matrix_view<const T, IdxT, raft::row_major> queries,                \
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,                 \
              raft::device_matrix_view<T, IdxT, raft::row_major> distances);

CUVS_INST_BFKNN(float, int64_t);

}  // namespace cuvs::neighbors::brute_force