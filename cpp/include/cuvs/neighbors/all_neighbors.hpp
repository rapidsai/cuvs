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

#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <variant>

namespace cuvs::neighbors::all_neighbors {

/**
 * @brief Parameters used to build an all-neighbors knn graph (find nearest neighbors for all the
 * training vectors)
 */
namespace graph_build_params {

/** Specialized parameters utilizing IVF-PQ to build knn graph */
struct ivf_pq_params {
  cuvs::neighbors::ivf_pq::index_params build_params;
  cuvs::neighbors::ivf_pq::search_params search_params;
  float refinement_rate = 2.0;
};

using nn_descent_params = cuvs::neighbors::nn_descent::index_params;
}  // namespace graph_build_params

using GraphBuildParams =
  std::variant<graph_build_params::ivf_pq_params, graph_build_params::nn_descent_params>;

/**
 * @brief Parameters used to build an all-neighbors graph  (find nearest neighbors for all the
 * training vectors)
 *
 * graph_build_params: graph building parameters for the given graph building algorithm. defaults
 * to ivfpq.
 * n_nearest_clusters: number of nearest clusters each data point will be assigned to in
 * the batching algorithm
 * n_clusters: number of total clusters (aka batches) to split the data into. If set to 1, algorithm
 * creates an all-neighbors graph without batching
 * metric: metric type
 *
 */
struct all_neighbors_params {
  /** Parameters for knn graph building algorithm
   *
   * Set ivf_pq_params, or nn_descent_params to select the graph build
   * algorithm and control their parameters.
   *
   * @code{.cpp}
   * all_neighbors::index_params params;
   * // 1. Choose IVF-PQ algorithm
   * params.graph_build_params = all_neighbors::graph_build_params::ivf_pq_params{};
   *
   * // 2. Choose NN Descent algorithm for kNN graph construction
   * params.graph_build_params = all_neighbors::graph_build_params::nn_descent_params{};
   *
   * @endcode
   */
  GraphBuildParams graph_build_params;

  /**
   * Usage of n_nearest_clusters and n_clusters
   *
   * The ratio of n_nearest_clusters / n_clusters determines device memory usage.
   * Approximately (n_nearest_clusters / n_clusters) * num_rows_in_entire_data number of rows will
   * be put on device memory at once.
   * E.g. between (n_nearest_clusters / n_clusters) = 2/10 and 2/20, the latter will use less device
   * memory.
   *
   * Larger n_nearest_clusters results in better accuracy of the final all-neighbors knn
   * graph. E.g. With the similar device memory usages, (n_nearest_clusters / n_clusters) = 4/20
   * will have better accuracy than 2/10 at the cost of performance.
   */
  size_t n_nearest_clusters           = 2;
  size_t n_clusters                   = 1;  // defaults to not batching
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

/**
 * @brief Builds an approximate all-neighbors knn graph  (find nearest neighbors for all the
 * training vectors)
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   all_neighbors::all_neighbors_params params;
 *  auto indices = raft::make_device_matrix<int64_t, int64_t>(handle, n_row, k);
 *  auto distances = raft::make_device_matrix<float, int64_t>(handle, n_row, k);
 *   all_neighbors::build(res, params, dataset, indices.view(), distances.view());
 * @endcode
 *
 * @param[in] handle raft::resources is an object mangaging resources
 * @param[in] params an instance of all_neighbors::all_neighbors_params that are parameters
 *               to build all-neighbors knn graph
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[out] indices nearest neighbor indices of shape [n_row x k]
 * @param[out] distances nearest neighbor distances [n_row x k]
 */
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::host_matrix_view<const float, int64_t, row_major> dataset,
  raft::device_matrix_view<int64_t, int64_t, row_major> indices,
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances = std::nullopt);

/**
 * @brief Builds an approximate all-neighbors knn graph (find nearest neighbors for all the training
 * vectors) params.n_clusters should be 1 for data on device. To use a larger params.n_clusters for
 * efficient device memory usage, put data on host RAM.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   all_neighbors::all_neighbors_params params;
 *  auto indices = raft::make_device_matrix<int64_t, int64_t>(handle, n_row, k);
 *  auto distances = raft::make_device_matrix<float, int64_t>(handle, n_row, k);
 *   all_neighbors::build(res, params, dataset, indices.view(), distances.view());
 * @endcode
 *
 * @param[in] handle raft::resources is an object mangaging resources
 * @param[in] params an instance of all_neighbors::all_neighbors_params that are parameters
 *               to build all-neighbors knn graph
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[out] indices nearest neighbor indices of shape [n_row x k]
 * @param[out] distances nearest neighbor distances [n_row x k]
 */
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::device_matrix_view<const float, int64_t, row_major> dataset,
  raft::device_matrix_view<int64_t, int64_t, row_major> indices,
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances = std::nullopt);
}  // namespace cuvs::neighbors::all_neighbors
