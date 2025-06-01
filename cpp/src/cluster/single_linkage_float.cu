/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "single_linkage.cuh"
#include <cuvs/cluster/agglomerative.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::cluster::agglomerative {

void single_linkage(raft::resources const& handle,
                    raft::device_matrix_view<const float, int, raft::row_major> X,
                    raft::device_matrix_view<int, int, raft::row_major> dendrogram,
                    raft::device_vector_view<int, int> labels,
                    cuvs::distance::DistanceType metric,
                    size_t n_clusters,
                    cuvs::cluster::agglomerative::Linkage linkage,
                    std::optional<int> c)
{
  if (linkage == Linkage::KNN_GRAPH) {
    single_linkage<float, int, Linkage::KNN_GRAPH>(
      handle, X, dendrogram, labels, metric, n_clusters, c);
  } else {
    single_linkage<float, int, Linkage::PAIRWISE>(
      handle, X, dendrogram, labels, metric, n_clusters, c);
  }
}

namespace helpers {
void build_linkage(raft::resources const& handle,
                   raft::device_matrix_view<const float, int, raft::row_major> X,
                   cuvs::distance::DistanceType metric,
                   raft::device_coo_matrix_view<float, int, int, int> out_mst,
                   raft::device_matrix_view<int, int> dendrogram,
                   raft::device_vector_view<float, int> out_distances,
                   raft::device_vector_view<int, int> out_sizes,
                   std::optional<raft::device_vector_view<float, int>> core_dists)
{
  /**
   * Construct MST sorted by weights
   */
  if (core_dists.has_value()) {
    RAFT_EXPECTS(core_dists.extent(0) == static_cast<size_t>(X.extent(0)),
                 "core_dists doesn't have expected size");
    RAFT_EXPECTS(indptr.extent(0) == static_cast<size_t>(X.extent(0) + 1),
                 "indptr doesn't have expected size");
    detail::build_mr_linkage<float, int, int>(handle,
                                              X,
                                              metric,
                                              min_samples,
                                              core_dists.value(),
                                              out_mst,
                                              dendrogram,
                                              out_distances,
                                              out_sizes)
  } else {
    detail::build_pw_linkage<float, int, int>(
      handle, X, metric, c, out_mst, dendrogram, out_distances, out_sizes);
  }
}
}  // namespace helpers
}  // namespace cuvs::cluster::agglomerative
