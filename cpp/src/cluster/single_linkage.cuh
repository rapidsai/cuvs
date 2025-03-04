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

#include "detail/single_linkage.cuh"
#include <cuvs/cluster/agglomerative.hpp>
#include <raft/core/device_mdspan.hpp>

namespace cuvs::cluster::agglomerative {

/**
 * Note: All of the functions below in the  cuvs::cluster namespace are deprecated
 * and will be removed in a future release. Please use   cuvs::cluster::agglomerative
 * instead.
 */

/**
 * Single-linkage clustering, capable of constructing a KNN graph to
 * scale the algorithm beyond the n^2 memory consumption of implementations
 * that use the fully-connected graph of pairwise distances by connecting
 * a knn graph when k is not large enough to connect it.

 * @tparam value_idx
 * @tparam value_t
 * @tparam dist_type method to use for constructing connectivities graph
 * @param[in] handle raft handle
 * @param[in] X dense input matrix in row-major layout
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metrix to use when constructing connectivities graph
 * @param[out] out struct containing output dendrogram and cluster assignments
 * @param[in] c a constant used when constructing connectivities from knn graph. Allows the indirect
 control
 *            of k. The algorithm will set `k = log(n) + c`
 * @param[in] n_clusters number of clusters to assign data samples
 */
template <typename value_idx, typename value_t, Linkage dist_type = Linkage::KNN_GRAPH>
void single_linkage(raft::resources const& handle,
                    const value_t* X,
                    value_idx m,
                    value_idx n,
                    cuvs::distance::DistanceType metric,
                    single_linkage_output<value_idx>* out,
                    int c,
                    size_t n_clusters)
{
  detail::single_linkage<value_idx, value_t, dist_type>(
    handle, X, m, n, metric, out, c, n_clusters);
}

/**
 * Single-linkage clustering, capable of constructing a KNN graph to
 * scale the algorithm beyond the n^2 memory consumption of implementations
 * that use the fully-connected graph of pairwise distances by connecting
 * a knn graph when k is not large enough to connect it.

 * @tparam value_idx
 * @tparam value_t
 * @tparam dist_type method to use for constructing connectivities graph
 * @param[in] handle raft handle
 * @param[in] X dense input matrix in row-major layout
 * @param[out] dendrogram output dendrogram (size [n_rows - 1] * 2)
 * @param[out] labels output labels vector (size n_rows)
 * @param[in] metric distance metrix to use when constructing connectivities graph
 * @param[in] n_clusters number of clusters to assign data samples
 * @param[in] c a constant used when constructing connectivities from knn graph. Allows the indirect
 control of k. The algorithm will set `k = log(n) + c`
 */
template <typename value_t, typename idx_t, Linkage dist_type = Linkage::KNN_GRAPH>
void single_linkage(raft::resources const& handle,
                    raft::device_matrix_view<const value_t, idx_t, raft::row_major> X,
                    raft::device_matrix_view<idx_t, idx_t, raft::row_major> dendrogram,
                    raft::device_vector_view<idx_t, idx_t> labels,
                    cuvs::distance::DistanceType metric,
                    size_t n_clusters,
                    std::optional<int> c = std::make_optional<int>(DEFAULT_CONST_C))
{
  single_linkage_output<idx_t> out_arrs;
  out_arrs.children = dendrogram.data_handle();
  out_arrs.labels   = labels.data_handle();

  single_linkage<idx_t, value_t, dist_type>(handle,
                                            X.data_handle(),
                                            X.extent(0),
                                            X.extent(1),
                                            metric,
                                            &out_arrs,
                                            c.has_value() ? c.value() : DEFAULT_CONST_C,
                                            n_clusters);
}
};  // namespace   cuvs::cluster::agglomerative
