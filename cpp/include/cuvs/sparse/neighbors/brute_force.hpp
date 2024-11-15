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

#include <cuvs/distance/distance.hpp>

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>

namespace cuvs::sparse::neighbors::brute_force {

/**
 * @defgroup sparse_bruteforce_cpp_index Sparse Brute Force index
 * @{
 */
/**
 * @brief Sparse Brute Force index.
 *
 * @tparam T Data element type
 * @tparam IdxT Index element type
 */
template <typename T, typename IdxT>
struct index {
 public:
  index(const index&)            = delete;
  index(index&&)                 = default;
  index& operator=(const index&) = delete;
  index& operator=(index&&)      = default;
  ~index()                       = default;

  /** Construct a sparse brute force index from dataset */
  index(raft::resources const& res,
        raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset,
        cuvs::distance::DistanceType metric,
        T metric_arg);

  /** Distance metric used for retrieval */
  cuvs::distance::DistanceType metric() const noexcept { return metric_; }

  /** Metric argument */
  T metric_arg() const noexcept { return metric_arg_; }

  raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset() const noexcept
  {
    return dataset_;
  }

 private:
  raft::device_csr_matrix_view<const T, IdxT, IdxT, IdxT> dataset_;
  cuvs::distance::DistanceType metric_;
  T metric_arg_;
};

/**
 * @}
 */

/**
 * @defgroup sparse_bruteforce_cpp_index_build Bruteforce index build
 * @{
 */

/*
 * @brief Build the Sparse index from the dataset
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // create and fill the index from a CSR dataset
 *   auto index = brute_force::build(handle, dataset, metric);
 * @endcode
 *
 * @param[in] handle
 * @param[in] dataset A sparse CSR matrix in device memory to search against
 * @param[in] metric cuvs::distance::DistanceType
 * @param[in] metric_arg metric argument
 *
 * @return the constructed Sparse brute-force index
 */
auto build(raft::resources const& handle,
           raft::device_csr_matrix_view<const float, int, int, int> dataset,
           cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Unexpanded,
           float metric_arg = 0) -> cuvs::sparse::neighbors::brute_force::index<float, int>;
/**
 * @}
 */

/**
 * @defgroup sparse_bruteforce_cpp_index_search Sparse Brute Force index search
 * @{
 */

struct search_params {
  int batch_size_index = 2 << 14;
  int batch_size_query = 2 << 14;
};

/*
 * @brief Search the sparse bruteforce index for nearest neighbors
 *
 * @param[in] handle
 * @param[in] index Sparse brute-force constructed index
 * @param[in] queries a sparse CSR matrix on the device to query
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 */
void search(raft::resources const& handle,
            const cuvs::sparse::neighbors::brute_force::search_params& params,
            const cuvs::sparse::neighbors::brute_force::index<float, int>& index,
            raft::device_csr_matrix_view<const float, int, int, int> dataset,
            raft::device_matrix_view<int, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances);
/**
 * @}
 */
}  // namespace cuvs::sparse::neighbors::brute_force
