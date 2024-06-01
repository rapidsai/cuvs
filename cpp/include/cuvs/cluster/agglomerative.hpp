/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <optional>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::agglomerative {

// constant to indirectly control the number of neighbors. k = sqrt(n) + c. default to 15
constexpr int DEFAULT_CONST_C = 15;

/**
 * @defgroup agglomerative_params agglomerative clustering hyperparameters
 * @{
 */

/**
 * Determines the method for computing the minimum spanning tree (MST)
 */
enum Linkage {

  /**
   * Use a pairwise distance matrix as input to the mst. This
   * is very fast and the best option for fairly small datasets (~50k data points)
   */
  PAIRWISE = 0,

  /**
   * Construct a KNN graph as input to the mst and provide additional
   * edges if the mst does not converge. This is slower but scales
   * to very large datasets.
   */
  KNN_GRAPH = 1
};

/**
 * @}
 */

/**
 * Simple container object for consolidating linkage results. This closely
 * mirrors the trained instance variables populated in
 * Scikit-learn's AgglomerativeClustering estimator.
 * @tparam idx_t
 */
template <typename idx_t>
class single_linkage_output {
 public:
  idx_t m;
  idx_t n_clusters;

  idx_t n_leaves;
  idx_t n_connected_components;

  // TODO: These will be made private in a future release
  idx_t* labels;    // size: m
  idx_t* children;  // size: (m-1, 2)

  raft::device_vector_view<idx_t> get_labels()
  {
    return raft::make_device_vector_view<idx_t>(labels, m);
  }

  raft::device_matrix_view<idx_t> get_children()
  {
    return raft::make_device_matrix_view<idx_t>(children, m - 1, 2);
  }
};

/**
 * @defgroup single_linkage single-linkage clustering APIs
 * @{
 */
/**
 * Single-linkage clustering, capable of constructing a KNN graph to
 * scale the algorithm beyond the n^2 memory consumption of implementations
 * that use the fully-connected graph of pairwise distances by connecting
 * a knn graph when k is not large enough to connect it.

 * @param[in] handle raft handle
 * @param[in] X dense input matrix in row-major layout
 * @param[out] dendrogram output dendrogram (size [n_rows - 1] * 2)
 * @param[out] labels output labels vector (size n_rows)
 * @param[in] metric distance metrix to use when constructing connectivities graph
 * @param[in] n_clusters number of clusters to assign data samples
 * @param[in] linkage strategy for constructing the linkage. PAIRWISE uses more memory but can be
 faster for
 *                    smaller datasets. KNN_GRAPH allows the memory usage to be controlled (using
 parameter c)
 *                    at the expense of potentially additional minimum spanning tree iterations.
 * @param[in] c a constant used when constructing linkage from knn graph. Allows the indirect
 control of k. The algorithm will set `k = log(n) + c`
 */
void single_linkage(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int, raft::row_major> X,
  raft::device_matrix_view<int, int, raft::row_major> dendrogram,
  raft::device_vector_view<int, int> labels,
  cuvs::distance::DistanceType metric,
  size_t n_clusters,
  cuvs::cluster::agglomerative::Linkage linkage = cuvs::cluster::agglomerative::Linkage::KNN_GRAPH,
  std::optional<int> c                          = std::make_optional<int>(DEFAULT_CONST_C));

/**
 * @}
 */
};  // end namespace  cuvs::cluster::agglomerative
