/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sparse/neighbors/cross_component_nn.cuh"
#include "raft/core/device_mdspan.hpp"
#include "raft/core/operators.hpp"
#include <cuvs/distance/distance.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/gather.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/sparse/solver/mst.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace cuvs::cluster::agglomerative::detail {

template <typename ValueIdx, typename ValueT>  // NOLINT(readability-identifier-naming)
void merge_msts(raft::sparse::solver::Graph_COO<ValueIdx, ValueIdx, ValueT>& coo1,
                raft::sparse::solver::Graph_COO<ValueIdx, ValueIdx, ValueT>& coo2,
                cudaStream_t stream)
{
  /** Add edges to existing mst **/
  int final_nnz = coo2.n_edges + coo1.n_edges;

  coo1.src.resize(final_nnz, stream);
  coo1.dst.resize(final_nnz, stream);
  coo1.weights.resize(final_nnz, stream);

  /**
   * Construct final edge list
   */
  raft::copy_async(coo1.src.data() + coo1.n_edges, coo2.src.data(), coo2.n_edges, stream);
  raft::copy_async(coo1.dst.data() + coo1.n_edges, coo2.dst.data(), coo2.n_edges, stream);
  raft::copy_async(coo1.weights.data() + coo1.n_edges, coo2.weights.data(), coo2.n_edges, stream);

  coo1.n_edges = final_nnz;
}

/**
 * Connect an unconnected knn graph (one in which mst returns an msf). The
 * device buffers underlying the Graph_COO object are modified in-place.
 * @tparam ValueIdx index type
 * @tparam ValueT floating-point value type
 * @param[in] handle raft handle
 * @param[in] X original dense data on device memory from which knn graph was constructed
 * @param[inout] msf edge list containing the mst result
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[inout] color the color labels array returned from the mst invocation
 * @return updated MST edge list
 */
template <typename ValueIdx,
          typename ValueT,
          typename RedOp>  // NOLINT(readability-identifier-naming)
void connect_knn_graph(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueT, ValueIdx> X,
  raft::sparse::solver::Graph_COO<ValueIdx, ValueIdx, ValueT>& msf,
  size_t m,
  size_t n,
  ValueIdx* color,
  RedOp reduction_op,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  raft::sparse::COO<ValueT, ValueIdx> connected_edges(stream);

  // default row and column batch sizes are chosen for computing cross component nearest neighbors.
  // Reference: PR #1445
  static constexpr size_t kDefaultRowBatchSize = 4096;
  static constexpr size_t kDefaultColBatchSize = 16;

  cuvs::sparse::neighbors::cross_component_nn<ValueIdx, ValueT>(handle,
                                                                connected_edges,
                                                                X.data_handle(),
                                                                color,
                                                                m,
                                                                n,
                                                                reduction_op,
                                                                min(m, kDefaultRowBatchSize),
                                                                min(n, kDefaultColBatchSize));

  rmm::device_uvector<ValueIdx> indptr2(m + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    connected_edges.rows(), connected_edges.nnz, indptr2.data(), m + 1, stream);

  // On the second call, we hand the MST the original colors
  // and the new set of edges and let it restart the optimization process
  auto new_mst =
    raft::sparse::solver::mst<ValueIdx, ValueIdx, ValueT, double>(handle,
                                                                  indptr2.data(),
                                                                  connected_edges.cols(),
                                                                  connected_edges.vals(),
                                                                  m,
                                                                  connected_edges.nnz,
                                                                  color,
                                                                  stream,
                                                                  false,
                                                                  false);

  merge_msts<ValueIdx, ValueT>(msf, new_mst, stream);
}

/**
 * Connect an unconnected knn graph (one in which mst returns an msf). The
 * device buffers underlying the Graph_COO object are modified in-place.
 * @tparam ValueIdx index type
 * @tparam ValueT floating-point value type
 * @param[in] handle raft handle
 * @param[in] X original dense data on host memory from which knn graph was constructed
 * @param[inout] msf edge list containing the mst result
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[inout] color the color labels array returned from the mst invocation
 * @return updated MST edge list
 */
template <typename ValueIdx,
          typename ValueT,
          typename RedOp = raft::identity_op>  // NOLINT(readability-identifier-naming)
void connect_knn_graph(
  raft::resources const& handle,
  raft::host_matrix_view<const ValueT, ValueIdx> X,
  raft::sparse::solver::Graph_COO<ValueIdx, ValueIdx, ValueT>& msf,
  size_t m,
  size_t n,
  ValueIdx* color,
  RedOp reduction_op                  = RedOp{},
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded)
{
  using cuvs::sparse::neighbors::cross_component_nn;
  using cuvs::sparse::neighbors::fix_connectivities_red_op;
  using cuvs::sparse::neighbors::get_n_components;
  using cuvs::sparse::neighbors::mutual_reachability_fix_connectivities_red_op;
  static_assert(
    std::is_same_v<RedOp, raft::identity_op> ||
      std::is_same_v<RedOp, mutual_reachability_fix_connectivities_red_op<ValueIdx, ValueT>> ||
      std::is_same_v<RedOp, fix_connectivities_red_op<ValueIdx, ValueT>>,
    "reduction_op must be identity_op, mutual_reachability_fix_connectivities_red_op, or "
    "fix_connectivities_red_op");

  auto stream      = raft::resource::get_cuda_stream(handle);
  int n_components = get_n_components(color, m, stream);

  rmm::device_uvector<ValueIdx> d_color_remapped(m, stream);
  raft::label::make_monotonic(d_color_remapped.data(), color, m, stream, true);

  std::vector<ValueIdx> h_color(m);
  raft::copy(h_color.data(), d_color_remapped.data(), m, stream);
  raft::resource::sync_stream(handle, stream);

  // make key (color) : value (vector of ids that have that color)
  std::unordered_map<ValueIdx, std::vector<ValueIdx>> component_map;
  for (ValueIdx i = 0; i < static_cast<ValueIdx>(m); ++i) {
    component_map[h_color[i]].push_back(i);
  }

  std::vector<std::tuple<ValueIdx, ValueIdx, ValueT>> selected_edges;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis;

  std::vector<ValueIdx> host_u_indices;
  std::vector<ValueIdx> host_v_indices;

  // connect i-1 component and i component
  for (int i = 1; i < n_components; ++i) {
    ValueIdx color_a = i - 1;
    ValueIdx color_b = i;

    const auto& nodes_a = component_map[color_a];
    const auto& nodes_b = component_map[color_b];

    // Randomly pick a data index from each component
    dis.param(std::uniform_int_distribution<>::param_type(0, nodes_a.size() - 1));
    ValueIdx u = nodes_a[dis(gen)];

    dis.param(std::uniform_int_distribution<>::param_type(0, nodes_b.size() - 1));
    ValueIdx v = nodes_b[dis(gen)];

    host_u_indices.push_back(u);
    host_v_indices.push_back(v);
  }

  size_t new_nnz = n_components - 1;

  auto device_u_indices = raft::make_device_vector<ValueIdx, ValueIdx>(handle, new_nnz);
  auto device_v_indices = raft::make_device_vector<ValueIdx, ValueIdx>(handle, new_nnz);

  raft::copy(device_u_indices.data_handle(), host_u_indices.data(), new_nnz, stream);
  raft::copy(device_v_indices.data_handle(), host_v_indices.data(), new_nnz, stream);

  auto data_u = raft::make_device_matrix<ValueT, ValueIdx>(handle, new_nnz, n);
  auto data_v = raft::make_device_matrix<ValueT, ValueIdx>(handle, new_nnz, n);

  raft::matrix::detail::gather(
    handle, X, raft::make_const_mdspan(device_u_indices.view()), data_u.view());
  raft::matrix::detail::gather(
    handle, X, raft::make_const_mdspan(device_v_indices.view()), data_v.view());

  auto pairwise_dist = raft::make_device_matrix<ValueT, ValueIdx>(handle, new_nnz, new_nnz);
  cuvs::distance::pairwise_distance(handle,
                                    raft::make_const_mdspan(data_u.view()),
                                    raft::make_const_mdspan(data_v.view()),
                                    pairwise_dist.view(),
                                    metric);

  auto pairwise_dist_vec = raft::make_device_vector<ValueT, ValueIdx>(handle, n_components - 1);
  raft::matrix::get_diagonal(
    handle, raft::make_const_mdspan(pairwise_dist.view()), pairwise_dist_vec.view());

  if constexpr (std::is_same<
                  RedOp,
                  mutual_reachability_fix_connectivities_red_op<ValueIdx, ValueT>>::value) {
    raft::linalg::map_offset(
      handle,
      pairwise_dist_vec.view(),
      [pairwise_dist_ptr    = pairwise_dist_vec.data_handle(),
       core_dist_ptr        = reduction_op.core_dists,
       device_u_indices_ptr = device_u_indices.data_handle(),
       device_v_indices_ptr = device_v_indices.data_handle()] __device__(auto i) -> float {
        float dist        = pairwise_dist_ptr[i];
        float u_core_dist = core_dist_ptr[device_u_indices_ptr[i]];
        float v_core_dist = core_dist_ptr[device_v_indices_ptr[i]];
        return fmaxf(dist, fmaxf(u_core_dist, v_core_dist));
      });
  }

  // sort in order of rows to run sorted_coo_to_csr
  auto rows_begin = thrust::device_pointer_cast(device_u_indices.data_handle());
  auto cols_begin = thrust::device_pointer_cast(device_v_indices.data_handle());
  auto dist_begin = thrust::device_pointer_cast(pairwise_dist_vec.data_handle());

  auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(cols_begin, dist_begin));
  thrust::sort_by_key(rows_begin, rows_begin + new_nnz, zipped_begin);

  rmm::device_uvector<ValueIdx> indptr2(m + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(
    device_u_indices.data_handle(), new_nnz, indptr2.data(), m + 1, stream);

  // On the second call, we hand the MST the original colors
  // and the new set of edges and let it restart the optimization process
  auto new_mst =
    raft::sparse::solver::mst<ValueIdx, ValueIdx, ValueT, double>(handle,
                                                                  indptr2.data(),
                                                                  device_v_indices.data_handle(),
                                                                  pairwise_dist_vec.data_handle(),
                                                                  m,
                                                                  new_nnz,
                                                                  color,
                                                                  stream,
                                                                  false,
                                                                  false);

  merge_msts<ValueIdx, ValueT>(msf, new_mst, stream);
}

/**
 * Constructs an MST and sorts the resulting edges in ascending
 * order by their weight.
 *
 * Hierarchical clustering heavily relies upon the ordering
 * and vertices returned in the MST. If the result of the
 * MST was actually a minimum-spanning forest, the CSR
 * being passed into the MST is not connected. In such a
 * case, this graph will be connected by performing a
 * KNN across the components.
 * @tparam ValueIdx
 * @tparam ValueT
 * @param[in] handle raft handle
 * @param[in] X dataset residing on host or device memory
 * @param[in] indptr CSR indptr of connectivities graph
 * @param[in] indices CSR indices array of connectivities graph
 * @param[in] pw_dists CSR weights array of connectivities graph
 * @param[in] m number of rows in X / src vertices in connectivities graph
 * @param[in] n number of columns in X
 * @param[out] mst_src output src edges
 * @param[out] mst_dst output dst edges
 * @param[out] mst_weight output weights (distances)
 * @param[in] max_iter maximum iterations to run knn graph connection. This
 *  argument is really just a safeguard against the potential for infinite loops.
 */
template <typename ValueIdx,
          typename ValueT,
          typename RedOp>  // NOLINT(readability-identifier-naming)
void build_sorted_mst(
  raft::resources const& handle,
  const ValueT* X,
  const ValueIdx* indptr,
  const ValueIdx* indices,
  const ValueT* pw_dists,
  size_t m,
  size_t n,
  ValueIdx* mst_src,
  ValueIdx* mst_dst,
  ValueT* mst_weight,
  ValueIdx* color,
  size_t nnz,
  RedOp reduction_op,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded,
  int max_iter                        = 10)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  // We want to have MST initialize colors on first call.
  auto mst_coo = raft::sparse::solver::mst<ValueIdx, ValueIdx, ValueT, double>(
    handle, indptr, indices, pw_dists, static_cast<ValueIdx>(m), nnz, color, stream, false, true);

  int iters        = 1;
  int n_components = cuvs::sparse::neighbors::get_n_components(color, m, stream);

  bool data_on_device = raft::memory_type_from_pointer(X) != raft::memory_type::host;

  while (n_components > 1 && iters < max_iter) {
    if (data_on_device) {
      connect_knn_graph<ValueIdx, ValueT>(
        handle,
        raft::make_device_matrix_view<const ValueT, ValueIdx>(X, m, n),
        mst_coo,
        m,
        n,
        color,
        reduction_op);
    } else {
      connect_knn_graph<ValueIdx, ValueT>(
        handle,
        raft::make_host_matrix_view<const ValueT, ValueIdx>(X, m, n),
        mst_coo,
        m,
        n,
        color,
        reduction_op,
        metric);
    }

    iters++;

    n_components = cuvs::sparse::neighbors::get_n_components(color, m, stream);
  }

  /**
   * The `max_iter` argument was introduced only to prevent the potential for an infinite loop.
   * Ideally the log2(n) guarantees of the MST should be enough to connect KNN graphs with a
   * massive number of data samples in very few iterations. If it does not, there are 3 likely
   * reasons why (in order of their likelihood):
   * 1. There is a bug in this code somewhere
   * 2. Either the given KNN graph wasn't generated from X or the same metric is not being used
   *    to generate the 1-nn (currently only L2SqrtExpanded is supported).
   * 3. max_iter was not large enough to connect the graph (less likely).
   *
   * Note that a KNN graph generated from 50 random isotropic balls (with significant overlap)
   * was able to be connected in a single iteration.
   */
  RAFT_EXPECTS(n_components == 1,
               "KNN graph could not be connected in %d iterations. "
               "Please verify that the input knn graph is generated from X "
               "(and the same distance metric used),"
               " or increase 'max_iter'",
               max_iter);

  raft::sparse::op::coo_sort_by_weight(
    mst_coo.src.data(), mst_coo.dst.data(), mst_coo.weights.data(), mst_coo.n_edges, stream);

  raft::copy_async(mst_src, mst_coo.src.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_dst, mst_coo.dst.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_weight, mst_coo.weights.data(), mst_coo.n_edges, stream);
}

};  // namespace  cuvs::cluster::agglomerative::detail
