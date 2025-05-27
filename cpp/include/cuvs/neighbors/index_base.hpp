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

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <memory>
#include <vector>

namespace cuvs::neighbors {

/**
 * @brief Common interface for ANN index implementations.
 *
 * Provides a unified abstraction for different approximate nearest neighbor (ANN) index types,
 * enabling polymorphic access to core operations such as build, search, size, and metric.
 *
 * This interface allows:
 * - Polymorphic usage of index implementations across the library
 * - Easy integration of new index types via a consistent API
 * - Support for composite or hierarchical index structures
 * - Seamless compatibility between existing and future components
 *
 * @tparam T Data type of vectors (e.g., float, int8_t)
 * @tparam IdxT Index type for dataset rows
 * @tparam OutputIdxT Output index type (defaults to IdxT)
 */
template <typename T, typename IdxT, typename OutputIdxT = IdxT>
struct IndexBase {
  using value_type        = T;
  using index_type        = IdxT;
  using out_index_type    = OutputIdxT;
  using matrix_index_type = int64_t;

  // Don't allow copying the index for performance reasons
  IndexBase()                                    = default;
  IndexBase(const IndexBase&)                    = delete;
  IndexBase(IndexBase&&)                         = default;
  auto operator=(const IndexBase&) -> IndexBase& = delete;
  auto operator=(IndexBase&&) -> IndexBase&      = default;
  virtual ~IndexBase()                           = default;

  /* Future implementation:
  virtual void build(
    const raft::resources& handle,
    const cuvs::neighbors::build_params& params,
    raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> dataset) = 0;
  */

  /**
   * @brief Perform approximate nearest neighbor search.
   *
   * Searches the index for the k-nearest neighbors of each query point.
   * The number of neighbors to find is determined by the neighbors matrix extent.
   *
   * @param[in] handle CUDA resources for executing operations
   * @param[in] params Search parameters specific to the index implementation
   * @param[in] queries Matrix of query vectors to search for [n_queries, dim]
   * @param[out] neighbors Matrix to store neighbor indices [n_queries, k]
   * @param[out] distances Matrix to store distances to neighbors [n_queries, k]
   * @param[in] filter Optional filter to exclude certain vectors from search results
   */
  virtual void search(
    const raft::resources& handle,
    const cuvs::neighbors::search_params& params,
    raft::device_matrix_view<const value_type, matrix_index_type, raft::row_major> queries,
    raft::device_matrix_view<out_index_type, matrix_index_type, raft::row_major> neighbors,
    raft::device_matrix_view<float, matrix_index_type, raft::row_major> distances,
    const cuvs::neighbors::filtering::base_filter& filter =
      cuvs::neighbors::filtering::none_sample_filter{}) const = 0;

  /**
   * @brief Get the number of vectors in the index.
   *
   * @return Number of indexed vectors
   */
  virtual index_type size() const noexcept = 0;

  /**
   * @brief Get the distance metric used by the index.
   *
   * @return Distance metric type (e.g., L2, InnerProduct)
   */
  virtual cuvs::distance::DistanceType metric() const noexcept = 0;
};

}  // namespace cuvs::neighbors
