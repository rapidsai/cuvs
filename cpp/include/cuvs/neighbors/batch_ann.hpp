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

namespace cuvs::neighbors::batch_ann {

/**
 * @brief batch_ann Build an approximate nearest neighbors index using the batching algorithm
 * The index contains an all-neighbors graph of the input dataset.
 * Indices are stored in host memory of dimensions (n_rows, k).
 * Distances are stored in device memory of dimensions(n_rows, k)
 *
 * @tparam IdxT dtype to be used for indices
 * @tparam DistT dtype to be used for distances
 */
template <typename IdxT, typename DistT = float>
struct index : cuvs::neighbors::index {
 public:
  /**
   * @brief Construct a new all-neighbors index object
   *
   * This constructor creates an all-neighbors knn-graph in host memory.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are (n_rows, k).
   *
   * @param res raft::resources is an object mangaging resources
   * @param n_rows number of rows in knn-graph
   * @param k number of nearest neighbors in knn-graph
   * @param return_distances whether to return distances
   */
  index(raft::resources const& res, int64_t n_rows, int64_t k, bool return_distances = false)
    : cuvs::neighbors::index(),
      res_{res},
      k_{k},
      graph_{raft::make_host_matrix<IdxT, IdxT, raft::row_major>(n_rows, k)},
      graph_view_{graph_.view()},
      return_distances_{return_distances}
  {
    if (return_distances) {
      distances_      = raft::make_device_matrix<DistT, IdxT>(res, n_rows, k);
      distances_view_ = distances_.value().view();
    }
  }

  /**
   * @brief Construct a new index object
   *
   * This constructor creates an all-neighbors graph using a user allocated host memory knn-graph.
   * The type of the knn-graph is a dense raft::host_matrix and dimensions are (n_rows, k).
   *
   * @param res raft::resources is an object mangaging resources
   * @param graph_view raft::host_matrix_view<IdxT, IdxT, raft::row_major> for storing knn-graph
   * @param distances_view optional raft::device_matrix_view<DistT, IdxT, row_major> for storing
   * distances
   */
  index(
    raft::resources const& res,
    raft::host_matrix_view<IdxT, IdxT, raft::row_major> graph_view,
    std::optional<raft::device_matrix_view<DistT, IdxT, row_major>> distances_view = std::nullopt)
    : cuvs::neighbors::index(),
      res_{res},
      k_{graph_view.extent(1)},
      graph_{raft::make_host_matrix<IdxT, IdxT, raft::row_major>(0, 0)},
      graph_view_{graph_view},
      distances_view_{distances_view},
      return_distances_{distances_view.has_value()}
  {
  }

  bool return_distances() { return return_distances_; }

  int64_t k() { return k_; }

  /** neighborhood graph [size, k] */
  [[nodiscard]] inline auto graph() noexcept -> raft::host_matrix_view<IdxT, IdxT, raft::row_major>
  {
    return graph_view_;
  }

  /** neighborhood graph distances [size, k] */
  [[nodiscard]] inline auto distances() noexcept
    -> std::optional<device_matrix_view<DistT, IdxT, row_major>>
  {
    return distances_view_;
  }

  // Don't allow copying the index for performance reasons (try avoiding copying data)
  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;
  ~index()                               = default;

 private:
  raft::resources const& res_;
  int64_t k_;
  bool return_distances_;
  raft::host_matrix<IdxT, IdxT, raft::row_major> graph_;
  std::optional<raft::device_matrix<DistT, IdxT, row_major>> distances_;
  raft::host_matrix_view<IdxT, IdxT, raft::row_major> graph_view_;
  std::optional<raft::device_matrix_view<DistT, IdxT, row_major>> distances_view_;
};

/**
 * @brief ANN parameters used by the batching algorithm to build an all-neighbors knn graph
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

/**
 * @brief Parameters used to build an all-neighbors graph using the batching algorithm
 *
 * graph_build_params: graph building parameters for the given graph building algorithm. defaults
 * to ivfpq
 * n_nearest_clusters: number of nearest clusters each data point will be assigned to in
 * the batching algorithm
 * n_clusters: number of total clusters (aka batches) to split the data into
 *
 */
struct index_params : cuvs::neighbors::index_params {
  /** Parameters for graph building for the batching algorithm.
   *
   * Set ivf_pq_params, or nn_descent_params to select the graph build
   * algorithm and control their parameters.
   *
   * @code{.cpp}
   * batch_ann::index_params params;
   * // 1. Choose IVF-PQ algorithm
   * params.graph_build_params = batch_ann::graph_build_params::ivf_pq_params{};
   *
   * // 2. Choose NN Descent algorithm for kNN graph construction
   * params.graph_build_params = batch_ann::graph_build_params::nn_descent_params{};
   *
   * @endcode
   */
  std::variant<graph_build_params::ivf_pq_params, graph_build_params::nn_descent_params>
    graph_build_params;

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
  size_t n_nearest_clusters = 2;
  size_t n_clusters         = 4;
};

auto build(const raft::resources& handle,
           raft::host_matrix_view<const float, int64_t, row_major> dataset,
           int64_t k,
           const index_params& params,
           bool return_distances = false) -> index<int64_t, float>;

void build(const raft::resources& handle,
           raft::host_matrix_view<const float, int64_t, row_major> dataset,
           const index_params& params,
           index<int64_t, float>& idx);

}  // namespace cuvs::neighbors::batch_ann
