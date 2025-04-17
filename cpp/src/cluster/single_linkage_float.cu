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
// #include <cuvs/cluster/agglomerative.hpp>

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

void build_mutual_reachability_linkage(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int, raft::row_major> X,
  cuvs::distance::DistanceType metric,
  raft::device_vector_view<float, int> core_dists,
  raft::device_vector_view<int, int> indptr,
  raft::sparse::COO<float, int>& mutual_reachability_coo,
  raft::device_vector_view<int, int> out_mst_src,
  raft::device_vector_view<int, int> out_mst_dst,
  raft::device_vector_view<float, int> out_mst_weights,
  raft::device_vector_view<int, int> out_children,
  raft::device_vector_view<float, int> out_deltas,
  raft::device_vector_view<int, int> out_sizes)
{
  build_mutual_reachability_linkage(handle,
                                    X,
                                    metric,
                                    core_dists,
                                    indptr,
                                    mutual_reachability_coo,
                                    out_mst_src,
                                    out_mst_dst,
                                    out_mst_weights,
                                    out_children,
                                    out_deltas,
                                    out_sizes);
}
}  // namespace cuvs::cluster::agglomerative
