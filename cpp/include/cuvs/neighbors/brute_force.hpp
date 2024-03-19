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
 * @defgroup bruteforce_cpp_index Bruteforce index
 * @{
 */
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

  /** Distance metric used for retrieval */
  cuvs::distance::DistanceType metric() const noexcept;

  /** Metric argument */
  T metric_arg() const noexcept;

  /** Total length of the index (number of vectors). */
  size_t size() const noexcept;

  /** Dimensionality of the data. */
  size_t dim() const noexcept;

  /** Dataset [size, dim] */
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept;

  /** Dataset norms */
  raft::device_vector_view<const T, int64_t, raft::row_major> norms() const;

  /** Whether ot not this index has dataset norms */
  bool has_norms() const noexcept;

  // Get pointer to underlying RAFT index, not meant to be used outside of cuVS
  inline const void* get_raft_index() const noexcept { return raft_index_.get(); }

 private:
  std::unique_ptr<void*> raft_index_;
};
/**
 * @}
 */

/**
 * @defgroup bruteforce_cpp_index_build Bruteforce index build
 * @{
 */
auto build(raft::resources const& res,
           raft::device_matrix_view<const float, int64_t, raft::row_major> dataset,
           cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
           float metric_arg                    = 0) -> cuvs::neighbors::brute_force::index<float>;
/**
 * @}
 */

/**
 * @defgroup bruteforce_cpp_index_search Bruteforce index search
 * @{
 */
void search(raft::resources const& res,
            const cuvs::neighbors::brute_force::index<float>& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
/**
 * @}
 */

}  // namespace cuvs::neighbors::brute_force
