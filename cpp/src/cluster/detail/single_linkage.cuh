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
  rmm::device_uvector<value_t> pw_dists(EMPTY, stream);

  /**
   * 1. Construct distance graph
   */
  detail::get_distance_graph<value_idx, value_t, dist_type>(
    handle, X, m, n, metric, indptr, indices, pw_dists, c);

  rmm::device_uvector<value_idx> mst_rows(m - 1, stream);
  rmm::device_uvector<value_idx> mst_cols(m - 1, stream);
  rmm::device_uvector<value_t> mst_data(m - 1, stream);

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
  detail::extract_flattened_clusters(handle, out->labels, out->children, n_clusters, m);

  out->m                      = m;
  out->n_clusters             = n_clusters;
  out->n_leaves               = m;
  out->n_connected_components = 1;
}

/**
 * Functor with reduction ops for performing fused 1-nn
 * computation in the mutual reachability space and guaranteeing only cross-component
 * neighbors are considered.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct FixConnectivitiesMRRedOp {
  value_t* core_dists;
  value_idx m;

  DI FixConnectivitiesMRRedOp() : m(0) {}

  FixConnectivitiesMRRedOp(value_t* core_dists_, value_idx m_) : core_dists(core_dists_), m(m_){};

  typedef typename raft::KeyValuePair<value_idx, value_t> KVP;
  DI void operator()(value_idx rit, KVP* out, const KVP& other) const
  {
    if (rit < m && other.value < std::numeric_limits<value_t>::max()) {
      value_t core_dist_rit   = core_dists[rit];
      value_t core_dist_other = max(core_dist_rit, max(core_dists[other.key], other.value));

      value_t core_dist_out;
      if (out->key > -1) {
        core_dist_out = max(core_dist_rit, max(core_dists[out->key], out->value));
      } else {
        core_dist_out = out->value;
      }

      bool smaller = core_dist_other < core_dist_out;
      out->key     = smaller ? other.key : out->key;
      out->value   = smaller ? core_dist_other : core_dist_out;
    }
  }

  DI KVP operator()(value_idx rit, const KVP& a, const KVP& b) const
  {
    if (rit < m && a.key > -1) {
      value_t core_dist_rit = core_dists[rit];
      value_t core_dist_a   = max(core_dist_rit, max(core_dists[a.key], a.value));

      value_t core_dist_b;
      if (b.key > -1) {
        core_dist_b = max(core_dist_rit, max(core_dists[b.key], b.value));
      } else {
        core_dist_b = b.value;
      }

      return core_dist_a < core_dist_b ? KVP(a.key, core_dist_a) : KVP(b.key, core_dist_b);
    }

    return b;
  }

  DI void init(value_t* out, value_t maxVal) const { *out = maxVal; }
  DI void init(KVP* out, value_t maxVal) const
  {
    out->key   = -1;
    out->value = maxVal;
  }

  DI void init_key(value_t& out, value_idx idx) const { return; }
  DI void init_key(KVP& out, value_idx idx) const { out.key = idx; }

  DI value_t get_value(KVP& out) const { return out.value; }
  DI value_t get_value(value_t& out) const { return out; }

  void gather(const raft::resources& handle, value_idx* map)
  {
    auto tmp_core_dists = raft::make_device_vector<value_t>(handle, m);
    thrust::gather(raft::resource::get_thrust_policy(handle),
                   map,
                   map + m,
                   core_dists,
                   tmp_core_dists.data_handle());
    raft::copy_async(
      core_dists, tmp_core_dists.data_handle(), m, raft::resource::get_cuda_stream(handle));
  }

  void scatter(const raft::resources& handle, value_idx* map)
  {
    auto tmp_core_dists = raft::make_device_vector<value_t>(handle, m);
    thrust::scatter(raft::resource::get_thrust_policy(handle),
                    core_dists,
                    core_dists + m,
                    map,
                    tmp_core_dists.data_handle());
    raft::copy_async(
      core_dists, tmp_core_dists.data_handle(), m, raft::resource::get_cuda_stream(handle));
  }
};

/**
 * Given a mutual reachability graph and sparse matrix, constructs a linkage over it by computing
 * mst and dendrogram. Returns mst edges sorted by weight
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle for resource reuse
 * @param[in] X data points (size m * n)
 * @param[in] m number of rows
 * @param[in] n number of columns
 * @param[in] metric distance metric to use
 * @param[in] params hyper parameters
 * @param[in] core_dists core distances (size m)
 * @param[out] out output container object
 * @param[out] out_dst children of output
 * @param[out] out_delta distances of output
 * @param[out] out_size cluster sizes of output
 */
template <typename value_idx = int, typename value_t = float, typename nnz_t>
void build_mutual_reachability_linkage(
  raft::resources const& handle,
  const value_t* X,
  size_t m,
  size_t n,
  cuvs::distance::DistanceType metric,
  value_t* core_dists,
  value_idx* indptr,
  raft::sparse::COO<value_t, value_idx, nnz_t>& mutual_reachability_coo,
  value_idx* out_mst_src,
  value_idx* out_mst_dst,
  value_t* out_mst_weights,
  value_idx* out_children,
  value_t* out_deltas,
  value_idx* out_sizes)
{
  /**
   * Construct MST sorted by weights
   */
  auto color = raft::make_device_vector<value_idx, value_idx>(handle, static_cast<value_idx>(m));
  FixConnectivitiesMRRedOp<value_idx, value_t> red_op(core_dists, static_cast<value_idx>(m));
  // during knn graph connection
  detail::build_sorted_mst(handle,
                           X,
                           indptr,
                           mutual_reachability_coo.cols(),
                           mutual_reachability_coo.vals(),
                           m,
                           n,
                           out_mst_src,
                           out_mst_dst,
                           out_mst_weights,
                           color.data_handle(),
                           mutual_reachability_coo.nnz,
                           red_op,
                           metric);

  /**
   * Perform hierarchical labeling
   */
  detail::build_dendrogram_host(
    handle, out_mst_src, out_mst_dst, out_mst_dst, m - 1, out_children, out_deltas, out_sizes);
}
};  // namespace  cuvs::cluster::agglomerative::detail
