/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "agglomerative.cuh"
#include "connectivities.cuh"
#include "mst.cuh"
#include <cuvs/cluster/agglomerative.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace cuvs::cluster::agglomerative::detail {

/**
 * Constructs a linkage by computing the minimum spanning tree and dendrogram. Returns mst edges sorted by weight and the dendrogram.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points (size m * n)
 * @param[in] m number of rows
 * @param[in] n number of columns
 * @param[in] metric distance metric to use
 * @param[in] min_samples this neighborhood will be selected for core distances
 * @param[in] core_dists core distances (size m)
 * @param[out] out_mst_src src vertex of MST edges (size m - 1)
 * @param[out] out_mst_dst dst vertex of MST eges (size m - 1)
 * @param[out] out_mst_weights weights of MST edges (size m - 1)
 * @param[out] out_children output dendrogram
 * @param[out] out_deltas distances of output
 * @param[out] out_sizes cluster sizes of output
 */
 template <typename value_idx = int, typename value_t = float, typename nnz_t>
 void build_linkage(
   raft::resources const& handle,
   const value_t* X,
   size_t m,
   size_t n,
   cuvs::distance::DistanceType metric,
   value_idx min_samples,
   value_idx c,
   value_t* core_dists,
   value_idx* out_mst_src,
   value_idx* out_mst_dst,
   value_t* out_mst_weights,
   value_idx* out_dendrogram,
   value_t* out_deltas,
   value_idx* out_sizes)
 {
   /**
    * Construct MST sorted by weights
    */
    RAFT_EXPECTS(core_dists.extent(0) == static_cast<size_t>(X.extent(0)),
               "core_dists doesn't have expected size");
  RAFT_EXPECTS(indptr.extent(0) == static_cast<size_t>(X.extent(0) + 1),
               "indptr doesn't have expected size");
  if (min_samples != 0 && core_dists != nullptr) {
  raft::sparse::COO<float> graph;
  cuvs::neighbors::detail::reachability::mutual_reachability_graph<int, float, size_t>(
    handle,
    X,
    m,
    n,
    metric,
    min_samples,
    1.0,
    indptr.data_handle(),
    core_dists.data_handle(),
    graph);
   auto color = raft::make_device_vector<value_idx, value_idx>(handle, static_cast<value_idx>(m));
    cuvs::sparse::neighbors::MutualReachabilityFixConnectivitiesRedOp<value_idx, value_t>
     reduction_op(core_dists, m);
   }

    raft::device_vector<value_idx> graph_indptr(m + 1);
    raft::device_coo_matrix<value_t, value_idx, value_idx, nnz_t> graph_coo,
   // during knn graph connection
   /**
   * 2. Construct MST, sorted by weights
   */
  rmm::device_uvector<value_idx> color(m, stream);
  cuvs::sparse::neighbors::FixConnectivitiesRedOp<value_idx, value_t> op(m);
  detail::build_sorted_mst<value_idx, value_t>(handle,
                                               X,
                                               indptr.data(),
                                               indices.data(),
                                               pw_dists.data(),
                                               m,
                                               n,
                                               mst_rows.data(),
                                               mst_cols.data(),
                                               mst_data.data(),
                                               color.data(),
                                               indices.size(),
                                               op,
                                               metric);

  pw_dists.release();

  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = mst_rows.size();

  rmm::device_uvector<value_t> out_delta(n_edges, stream);
  rmm::device_uvector<value_idx> out_size(n_edges, stream);
  // Create dendrogram
  detail::build_dendrogram_host<value_idx, value_t>(handle,
                                                    mst_rows.data(),
                                                    mst_cols.data(),
                                                    mst_data.data(),
                                                    n_edges,
                                                    out->children,
                                                    out_delta.data(),
                                                    out_size.data());
 }

static const size_t EMPTY = 0;

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
template <typename value_idx, typename value_t, Linkage dist_type>
void single_linkage(raft::resources const& handle,
                    const value_t* X,
                    size_t m,
                    size_t n,
                    cuvs::distance::DistanceType metric,
                    single_linkage_output<value_idx>* out,
                    int c,
                    size_t n_clusters)
{
  ASSERT(n_clusters <= m, "n_clusters must be less than or equal to the number of data points");

  auto stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> indptr(EMPTY, stream);
  rmm::device_uvector<value_idx> indices(EMPTY, stream);

  {
  rmm::device_uvector<value_t> pw_dists(EMPTY, stream);

  /**
   * 1. Construct distance graph
   */
  detail::get_distance_graph<value_idx, value_t, dist_type>(
    handle, X, m, n, metric, indptr, indices, pw_dists, c);
  }
  rmm::device_uvector<value_idx> mst_rows(m - 1, stream);
  rmm::device_uvector<value_idx> mst_cols(m - 1, stream);
  rmm::device_uvector<value_t> mst_data(m - 1, stream);

  build_linkage(handle,
                X,
                indptr.data(),
                graph.data(),
                m,
                n,
                mst_rows.data(),
              mst_cols.data(),
            mst_data.data(),
            metric);

  
  detail::extract_flattened_clusters(handle, out->labels, out->children, n_clusters, m);

  out->m                      = m;
  out->n_clusters             = n_clusters;
  out->n_leaves               = m;
  out->n_connected_components = 1;
}
};  // namespace  cuvs::cluster::agglomerative::detail
