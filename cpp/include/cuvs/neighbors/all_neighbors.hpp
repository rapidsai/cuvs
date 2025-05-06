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

// /**
//  * @brief Build an all-neighbors knn graph.
//  * The index contains an all-neighbors graph of the input dataset.
//  * Indices are stored in host memory of dimensions (n_rows, k).
//  * Distances are stored in device memory of dimensions(n_rows, k) if return_distances=true
//  *
//  * @tparam IdxT dtype to be used for indices
//  * @tparam DistT dtype to be used for distances
//  */
// template <typename IdxT, typename DistT = float>
// struct index : cuvs::neighbors::index {
//  public:
//   /**
//    * @brief Construct a new all-neighbors index object
//    *
//    * This constructor creates an all-neighbors knn graph in host memory.
//    * The type of the knn graph is a dense raft::host_matrix and dimensions are (n_rows, k).
//    *
//    * @param res raft::resources is an object mangaging resources
//    * @param n_rows number of rows in knn graph
//    * @param k number of nearest neighbors in knn graph
//    * @param return_distances bool for whether to return distances
//    */
//   index(raft::resources const& res, int64_t n_rows, int64_t k, bool return_distances = false)
//     : cuvs::neighbors::index(),
//       res_{res},
//       k_{k},
//       graph_{raft::make_host_matrix<IdxT, IdxT, row_major>(n_rows, k)},
//       graph_view_{graph_.view()},
//       return_distances_{return_distances}
//   {
//     if (return_distances) {
//       distances_.emplace(raft::make_device_matrix<DistT, IdxT>(res, n_rows, k));
//       distances_view_.emplace(distances_.value().view());
//     }
//   }

//   /**
//    * @brief Construct a new index object
//    *
//    * This constructor creates an all-neighbors graph using a user allocated host memory knn
//    graph.
//    * The type of the knn graph is a dense raft::host_matrix and dimensions are (n_rows, k).
//    *
//    * @param res raft::resources is an object mangaging resources
//    * @param graph_view raft::host_matrix_view<IdxT, IdxT> for storing knn graph
//    * @param distances_view optional raft::device_matrix_view<DistT, IdxT> for storing
//    * distances
//    */
//   index(
//     raft::resources const& res,
//     raft::host_matrix_view<IdxT, IdxT, row_major> graph_view,
//     std::optional<raft::device_matrix_view<DistT, IdxT, row_major>> distances_view =
//     std::nullopt) : cuvs::neighbors::index(),
//       res_{res},
//       k_{graph_view.extent(1)},
//       graph_{raft::make_host_matrix<IdxT, IdxT, row_major>(0, 0)},
//       graph_view_{graph_view},
//       distances_view_{distances_view},
//       return_distances_{distances_view.has_value()}
//   {
//   }

//   bool return_distances() { return return_distances_; }

//   int64_t k() { return k_; }

//   /** neighborhood graph [size, k] */
//   [[nodiscard]] inline auto graph() noexcept -> raft::host_matrix_view<IdxT, IdxT, row_major>
//   {
//     return graph_view_;
//   }

//   /** neighborhood graph distances [size, k] */
//   [[nodiscard]] inline auto distances() noexcept
//     -> std::optional<device_matrix_view<DistT, IdxT, row_major>>
//   {
//     return distances_view_;
//   }

//   // Don't allow copying the index for performance reasons (try avoiding copying data)
//   index(const index&)                    = delete;
//   index(index&&)                         = default;
//   auto operator=(const index&) -> index& = delete;
//   auto operator=(index&&) -> index&      = default;
//   ~index()                               = default;

//  private:
//   raft::resources const& res_;
//   int64_t k_;
//   bool return_distances_;
//   raft::host_matrix<IdxT, IdxT, row_major> graph_;
//   std::optional<raft::device_matrix<DistT, IdxT, row_major>> distances_;
//   raft::host_matrix_view<IdxT, IdxT, row_major> graph_view_;
//   std::optional<raft::device_matrix_view<DistT, IdxT, row_major>> distances_view_;
// };

/**
 * @brief Parameters used to build an all-neighbors knn graph
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
 * @brief Parameters used to build an all-neighbors graph
 *
 * graph_build_params: graph building parameters for the given graph building algorithm. defaults
 * to ivfpq.
 * n_nearest_clusters: number of nearest clusters each data point will be assigned to in
 * the batching algorithm
 * n_clusters: number of total clusters (aka batches) to split the data into. If set to 1, algorithm
 * creates an all-neighbors graph without batching
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
   * Hint1: the ratio of n_nearest_clusters / n_clusters determines device memory usage.
   * Approximately (n_nearest_clusters / n_clusters) * num_rows_in_entire_data number of rows will
   * be put on device memory at once.
   * E.g. between (n_nearest_clusters / n_clusters) = 2/10 and 2/20, the latter will use less device
   * memory.
   *
   * Hint2: larger n_nearest_clusters results in better accuracy of the final all-neighbors knn
   * graph. E.g. With the similar device memory usages, (n_nearest_clusters / n_clusters) = 4/20
   * will have better accuracy than 2/10 at the cost of performance.
   */
  size_t n_nearest_clusters           = 2;
  size_t n_clusters                   = 1;  // defaults to not batching
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

/**
 * @brief Builds a new approximate all-neighbors knn graph. Returns an approximate nearest neighbor
 * indices matrix and the corresponding distances (if return_distances=true) on the given dataset.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   all_neighbors::index_params index_params;
 *   // create and fill the index from a [N, D] raft::host_matrix_view dataset
 *   auto index = all_neighbors::build(res, dataset, k, index_params);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset.
 *   // index.distances() provides a raft::device_matrix_view of the corresponding distances
 * @endcode
 *
 * @param[in] handle raft::resources is an object mangaging resources
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[in] k number of nearest neighbors in the resulting knn graph
 * @param[in] params an instance of all_neighbors::index_params that are parameters
 *               to build all-neighbors knn graph
 * @param[in] return_distances boolean for whether to return the distances matrix as part of the
 * index
 * @return index<IdxT> index containing all-neighbors knn graph in host memory (and the
 * corresponding distances in device memory if return_distances = true)
 */
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::host_matrix_view<const float, int64_t, row_major> dataset,
  raft::device_matrix_view<int64_t, int64_t, row_major> indices,
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances = std::nullopt);

/**
 * @brief Builds a new approximate all-neighbors knn graph. Returns an approximate nearest neighbor
 * indices matrix and the corresponding distances (if distances_view has value in idx) on the given
 * dataset.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   all_neighbors::index_params index_params;
 *   //
 *   all_neighbors::index<IdxT, T> index{handle, dataset.extent(0), k, return_distances};
 *   // fill the index from a [N, D] raft::host_matrix_view dataset
 *   all_neighbors::build(res, dataset, index_params, index);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 *   // index.distances() provides a raft::device_matrix_view of the corresponding distances
 * @endcode
 *
 * @param[in] handle raft::resources is an object mangaging resources
 * @param[in] dataset raft::host_matrix_view input dataset expected to be located
 *                in host memory
 * @param[in] params an instance of all_neighbors::index_params that are parameters
 *               to build all-neighbors knn graph
 * @param[out] idx all_neighbors::index type holding the all-neighbors graph in host memory (and the
 * corresponding distances in device memory if return_distances = true)
 */
// void build(const raft::resources& handle,
//            raft::host_matrix_view<const float, int64_t, row_major> dataset,
//            const index_params& params,
//            index<int64_t, float>& idx);

/**
 * @brief Builds a new approximate all-neighbors knn graph. Returns an approximate nearest neighbor
 * indices matrix and the corresponding distances (if return_distances=true) on the given dataset.
 * params.n_clusters should be 1 for data on device. . To use a larger
 * params.n_clusters for efficient device memory usage, put data on host RAM.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   all_neighbors::index_params index_params;
 *   // create and fill the index from a [N, D] raft::device_matrix_view dataset
 *   auto index = all_neighbors::build(res, dataset, k, index_params);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset.
 *   // index.distances() provides a raft::device_matrix_view of the corresponding distances
 * @endcode
 *
 * @param[in] handle raft::resources is an object mangaging resources
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[in] k number of nearest neighbors in the resulting knn graph
 * @param[in] params an instance of all_neighbors::index_params that are parameters
 *               to build all-neighbors knn graph
 * @param[in] return_distances boolean for whether to return the distances matrix as part of the
 * index
 * @return index<IdxT> index containing all-neighbors knn graph in host memory (and the
 * corresponding distances in device memory if return_distances = true)
 */
void build(
  const raft::resources& handle,
  const all_neighbors_params& params,
  raft::device_matrix_view<const float, int64_t, row_major> dataset,
  raft::device_matrix_view<int64_t, int64_t, row_major> indices,
  std::optional<raft::device_matrix_view<float, int64_t, row_major>> distances = std::nullopt);

/**
 * @brief Builds a new approximate all-neighbors knn graph. Returns an approximate nearest neighbor
 * indices matrix and the corresponding distances (if distances_view has value in idx) on the given
 * dataset. params.n_clusters should be 1 for data on device. To use a larger params.n_clusters for
 * efficient device memory usage, put data on host RAM.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   all_neighbors::index_params index_params;
 *   //
 *   all_neighbors::index<IdxT, T> index{handle, dataset.extent(0), k, return_distances};
 *   // fill the index from a [N, D] raft::device_matrix_view dataset
 *   all_neighbors::build(res, dataset, index_params, index);
 *   // index.graph() provides a raft::host_matrix_view of an
 *   // all-neighbors knn graph of dimensions [N, k] of the input
 *   // dataset
 *   // index.distances() provides a raft::device_matrix_view of the corresponding distances
 * @endcode
 *
 * @param[in] handle raft::resources is an object mangaging resources
 * @param[in] dataset raft::device_matrix_view input dataset expected to be located
 *                in device memory
 * @param[in] params an instance of all_neighbors::index_params that are parameters
 *               to build all-neighbors knn graph
 * @param[out] idx all_neighbors::index type holding the all-neighbors graph in host memory (and the
 * corresponding distances in device memory if return_distances = true)
 */
// void build(const raft::resources& handle,
//            raft::device_matrix_view<const float, int64_t, row_major> dataset,
//            const index_params& params,
//            index<int64_t, float>& idx);

}  // namespace cuvs::neighbors::all_neighbors
