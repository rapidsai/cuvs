/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "detail/nn_descent.cuh"
#include "detail/nn_descent_batch.cuh"

#include <cmath>
#include <cstdint>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>

namespace cuvs::neighbors::nn_descent {

/**
 * @defgroup nn-descent CUDA gradient descent nearest neighbor
 * @{
 */

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - BitwiseHamming (when T == int8, uint8)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
template <typename T, typename IdxT = uint32_t>
auto build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset) -> index<IdxT>
{
  if (params.n_clusters > 1) {
    // related issue: https://github.com/rapidsai/cuvs/issues/1051
    RAFT_LOG_WARN(
      "NN Descent batch build (using n_clusters > 1) is deprecated and will be removed in a future "
      "release. Please use cuvs::all_neighbors::build(...) instead.");
    if constexpr (std::is_same_v<T, float>) {
      return detail::experimental::batch_build<T, IdxT>(res, params, dataset);
    } else {
      RAFT_FAIL("Batched nn-descent is only supported for float precision");
    }
  } else {
    return detail::build<T, IdxT>(res, params, dataset);
  }
}

/**
 * @brief Build nn-descent Index with dataset in device memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - BitwiseHamming (when T == int8, uint8)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(N, D);
 *   auto index = nn_descent::index{res, knn_graph.view()};
 *   nn_descent::build(res, index_params, dataset, index);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[out] idx  cuvs::neighbors::nn_descentindex containing all-neighbors knn graph
 * in host memory
 */
template <typename T, typename IdxT = uint32_t>
void build(raft::resources const& res,
           index_params const& params,
           raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,
           index<IdxT>& idx)
{
  if (params.n_clusters > 1) {
    // related issue: https://github.com/rapidsai/cuvs/issues/1051
    RAFT_LOG_WARN(
      "NN Descent batch build (using n_clusters > 1) is deprecated and will be removed in a future "
      "release. Please use cuvs::all_neighbors::build(...) instead.");
    if constexpr (std::is_same_v<T, float>) {
      detail::experimental::batch_build<T, IdxT>(res, params, dataset, idx);
    } else {
      RAFT_FAIL("Batched nn-descent is only supported for float precision");
    }
  } else {
    detail::build<T, IdxT>(res, params, dataset, idx);
  }
}

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - BitwiseHamming (when T == int8, uint8)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = nn_descent::build(res, index_params, dataset);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @return index<IdxT> index containing all-neighbors knn graph in host memory
 */
template <typename T, typename IdxT = uint32_t>
auto build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const T, int64_t, raft::row_major> dataset) -> index<IdxT>
{
  if (params.n_clusters > 1) {
    // related issue: https://github.com/rapidsai/cuvs/issues/1051
    RAFT_LOG_WARN(
      "NN Descent batch build (using n_clusters > 1) is deprecated and will be removed in a future "
      "release. Please use cuvs::all_neighbors::build(...) instead.");
    if constexpr (std::is_same_v<T, float>) {
      return detail::experimental::batch_build<T, IdxT>(res, params, dataset);
    } else {
      RAFT_FAIL("Batched nn-descent is only supported for float precision");
    }
  } else {
    return detail::build<T, IdxT>(res, params, dataset);
  }
}

/**
 * @brief Build nn-descent Index with dataset in host memory
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - L2SqrtExpanded
 * - CosineExpanded
 * - InnerProduct
 * - BitwiseHamming (when T == int8, uint8)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   nn_descent::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto knn_graph = raft::make_host_matrix<uint32_t, int64_t>(N, D);
 *   auto index = nn_descent::index{res, knn_graph.view()};
 *   nn_descent::build(res, index_params, dataset, index);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 * @endcode
 *
 * @tparam T data-type of the input dataset
 * @tparam IdxT data-type for the output index
 * @param[in] res raft::resources is an object mangaging resources
 * @param[in] params an instance of nn_descent::index_params that are parameters
 *               to run the nn-descent algorithm
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[out] idx  cuvs::neighbors::nn_descentindex containing all-neighbors knn graph
 * in host memory
 */
template <typename T, typename IdxT = uint32_t>
void build(raft::resources const& res,
           index_params const& params,
           raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,
           index<IdxT>& idx)
{
  if (params.n_clusters > 1) {
    // related issue: https://github.com/rapidsai/cuvs/issues/1051
    RAFT_LOG_WARN(
      "NN Descent batch build (using n_clusters > 1) is deprecated and will be removed in a future "
      "release. Please use cuvs::all_neighbors::build(...) instead.");
    if constexpr (std::is_same_v<T, float>) {
      detail::experimental::batch_build<T, IdxT>(res, params, dataset, idx);
    } else {
      RAFT_FAIL("Batched nn-descent is only supported for float precision");
    }
  } else {
    detail::build<T, IdxT>(res, params, dataset, idx);
  }
}

/** @} */  // end group nn-descent

}  // namespace cuvs::neighbors::nn_descent
