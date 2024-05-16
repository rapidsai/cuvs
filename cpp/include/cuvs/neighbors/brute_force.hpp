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
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdspan.hpp>

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

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.
   * This index will store a non-owning reference to the dataset.
   */
  index(raft::resources const& res,
        raft::host_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector<T, int64_t>>&& norms,
        cuvs::distance::DistanceType metric,
        T metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * Constructs a brute force index from a dataset. This lets us precompute norms for
   * the dataset, providing a speed benefit over doing this at query time.
   * The dataset will be copied to the device and the index will own the device memory.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector<T, int64_t>>&& norms,
        cuvs::distance::DistanceType metric,
        T metric_arg = 0.0);

  /** Construct a brute force index from dataset
   *
   * This class stores a non-owning reference to the dataset and norms here.
   * Having precomputed norms gives us a performance advantage at query time.
   */
  index(raft::resources const& res,
        raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view,
        std::optional<raft::device_vector_view<const T, int64_t>> norms_view,
        cuvs::distance::DistanceType metric,
        T metric_arg = 0.0);

  /**
   * Replace the dataset with a new dataset.
   */
  void update_dataset(raft::resources const& res,
                      raft::device_matrix_view<const T, int64_t, raft::row_major> dataset);

  /**
   * Replace the dataset with a new dataset.
   *
   * We create a copy of the dataset on the device. The index manages the lifetime of this copy.
   */
  void update_dataset(raft::resources const& res,
                      raft::host_matrix_view<const T, int64_t, raft::row_major> dataset);

  /** Distance metric used for retrieval */
  cuvs::distance::DistanceType metric() const noexcept { return metric_; }

  /** Metric argument */
  T metric_arg() const noexcept { return metric_arg_; }

  /** Total length of the index (number of vectors). */
  size_t size() const noexcept { return dataset_view_.extent(0); }

  /** Dimensionality of the data. */
  size_t dim() const noexcept { return dataset_view_.extent(1); }

  /** Dataset [size, dim] */
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset() const noexcept
  {
    return dataset_view_;
  }

  /** Dataset norms */
  raft::device_vector_view<const T, int64_t, raft::row_major> norms() const
  {
    return norms_view_.value();
  }

  /** Whether ot not this index has dataset norms */
  inline bool has_norms() const noexcept { return norms_view_.has_value(); }

 private:
  cuvs::distance::DistanceType metric_;
  raft::device_matrix<T, int64_t, raft::row_major> dataset_;
  std::optional<raft::device_vector<T, int64_t>> norms_;
  std::optional<raft::device_vector_view<const T, int64_t>> norms_view_;
  raft::device_matrix_view<const T, int64_t, raft::row_major> dataset_view_;
  T metric_arg_;
};
/**
 * @}
 */

/**
 * @defgroup bruteforce_cpp_index_build Bruteforce index build
 * @{
 */
/**
 * @brief Build the index from the dataset for efficient search.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = brute_force::build(handle, dataset, metric);
 * @endcode
 *
 * @param[in] handle
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 * @param[in] metric cuvs::distance::DistanceType
 * @param[in] metric_arg metric argument
 *
 * @return the constructed ivf-flat index
 */
auto build(raft::resources const& handle,
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
/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   brute_force::search(handle, index, queries1, out_inds1, out_dists1);
 *   brute_force::search(handle, index, queries2, out_inds2, out_dists2);
 *   brute_force::search(handle, index, queries3, out_inds3, out_dists3);
 *   ...
 * @endcode
 *
 * @param[in] handle
 * @param[in] index ivf-flat constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 */
void search(raft::resources const& handle,
            const cuvs::neighbors::brute_force::index<float>& index,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
/**
 * @}
 */

}  // namespace cuvs::neighbors::brute_force
