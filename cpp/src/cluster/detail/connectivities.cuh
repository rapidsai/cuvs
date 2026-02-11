/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors/detail/knn_graph.cuh"
#include "./kmeans_common.cuh"
#include <cuvs/cluster/agglomerative.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <limits>

namespace cuvs::cluster::agglomerative::detail {

template <Linkage dist_type,
          typename ValueIdx,
          typename value_t>  // NOLINT(readability-identifier-naming)
struct distance_graph_impl {
  void run(raft::resources const& handle,
           const value_t* X,
           ValueIdx m,
           ValueIdx n,
           cuvs::distance::DistanceType metric,
           rmm::device_uvector<ValueIdx>& indptr,
           rmm::device_uvector<ValueIdx>& indices,
           rmm::device_uvector<value_t>& data,
           int c);
};

/**
 * Connectivities specialization to build a knn graph
 * @tparam ValueIdx
 * @tparam value_t
 */
template <typename ValueIdx, typename value_t>  // NOLINT(readability-identifier-naming)
struct distance_graph_impl<Linkage::KNN_GRAPH, ValueIdx, value_t> {
  void run(raft::resources const& handle,
           const value_t* X,
           ValueIdx m,
           ValueIdx n,
           cuvs::distance::DistanceType metric,
           rmm::device_uvector<ValueIdx>& indptr,
           rmm::device_uvector<ValueIdx>& indices,
           rmm::device_uvector<value_t>& data,
           int c)
  {
    auto stream        = raft::resource::get_cuda_stream(handle);
    auto thrust_policy = raft::resource::get_thrust_policy(handle);

    // Need to symmetrize knn into undirected graph
    raft::sparse::COO<value_t, ValueIdx> knn_graph_coo(stream);

    auto x_view = raft::make_device_matrix_view<const value_t, ValueIdx, raft::row_major>(X, m, n);
    cuvs::neighbors::detail::knn_graph<ValueIdx, value_t, size_t>(
      handle, x_view, metric, knn_graph_coo, c);

    indices.resize(knn_graph_coo.nnz, stream);
    data.resize(knn_graph_coo.nnz, stream);

    // self-loops get max distance
    auto rows_view = raft::make_device_vector_view<const ValueIdx, ValueIdx>(knn_graph_coo.rows(),
                                                                             knn_graph_coo.nnz);
    auto cols_view = raft::make_device_vector_view<const ValueIdx, ValueIdx>(knn_graph_coo.cols(),
                                                                             knn_graph_coo.nnz);
    auto vals_in_view = raft::make_device_vector_view<const value_t, ValueIdx>(knn_graph_coo.vals(),
                                                                               knn_graph_coo.nnz);
    auto vals_out_view =
      raft::make_device_vector_view<value_t, ValueIdx>(knn_graph_coo.vals(), knn_graph_coo.nnz);

    raft::linalg::map(
      handle,
      vals_out_view,
      [=] __device__(const ValueIdx row, const ValueIdx col, const value_t val) -> value_t {
        bool self_loop = row == col;
        return (self_loop * std::numeric_limits<value_t>::max()) + (!self_loop * val);
      },
      rows_view,
      cols_view,
      vals_in_view);

    raft::sparse::convert::sorted_coo_to_csr(
      knn_graph_coo.rows(), knn_graph_coo.nnz, indptr.data(), m + 1, stream);

    // TODO(snanditale): Wouldn't need to copy here if we could compute knn
    // graph directly on the device uvectors
    // ref: https://github.com/rapidsai/raft/issues/227
    raft::copy_async(indices.data(), knn_graph_coo.cols(), knn_graph_coo.nnz, stream);
    raft::copy_async(data.data(), knn_graph_coo.vals(), knn_graph_coo.nnz, stream);
  }
};

template <typename ValueIdx>
RAFT_KERNEL fill_indices2(ValueIdx* indices, size_t m, size_t nnz)
{
  ValueIdx tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nnz) return;
  ValueIdx v   = tid % m;
  indices[tid] = v;
}

/**
 * Compute connected CSR of pairwise distances
 * @tparam ValueIdx
 * @tparam value_t
 * @param handle
 * @param X
 * @param m
 * @param n
 * @param metric
 * @param[out] indptr
 * @param[out] indices
 * @param[out] data
 */
template <typename ValueIdx, typename value_t>  // NOLINT(readability-identifier-naming)
void pairwise_distances(const raft::resources& handle,
                        const value_t* X,
                        ValueIdx m,
                        ValueIdx n,
                        cuvs::distance::DistanceType metric,
                        ValueIdx* indptr,
                        ValueIdx* indices,
                        value_t* data)
{
  auto stream      = raft::resource::get_cuda_stream(handle);
  auto exec_policy = raft::resource::get_thrust_policy(handle);

  ValueIdx nnz = m * m;

  ValueIdx blocks = raft::ceildiv(nnz, static_cast<ValueIdx>(256));
  fill_indices2<ValueIdx><<<blocks, 256, 0, stream>>>(indices, m, nnz);

  raft::linalg::map_offset(handle,
                           raft::make_device_vector_view<ValueIdx, ValueIdx>(indptr, m),
                           [=] __device__(ValueIdx idx) -> ValueIdx { return idx * m; });

  raft::update_device(indptr + m, &nnz, 1, stream);

  // TODO(snanditale): It would ultimately be nice if the MST could accept
  // dense inputs directly so we don't need to double the memory
  // usage to hand it a sparse array here.
  auto x_view = raft::make_device_matrix_view<const value_t, ValueIdx>(X, m, n);

  cuvs::cluster::kmeans::detail::pairwise_distance_kmeans<value_t, ValueIdx>(
    handle, x_view, x_view, raft::make_device_matrix_view<value_t, ValueIdx>(data, m, m), metric);

  // self-loops get max distance
  auto data_view = raft::make_device_vector_view<value_t, ValueIdx>(data, nnz);

  raft::linalg::map_offset(handle, data_view, [=] __device__(ValueIdx idx) -> value_t {
    value_t val    = data[idx];
    bool self_loop = idx % m == idx / m;
    return (self_loop * std::numeric_limits<value_t>::max()) + (!self_loop * val);
  });
}

/**
 * Connectivities specialization for pairwise distances
 * @tparam ValueIdx
 * @tparam value_t
 */
template <typename ValueIdx, typename value_t>  // NOLINT(readability-identifier-naming)
struct distance_graph_impl<Linkage::PAIRWISE, ValueIdx, value_t> {
  void run(const raft::resources& handle,
           const value_t* X,
           ValueIdx m,
           ValueIdx n,
           cuvs::distance::DistanceType metric,
           rmm::device_uvector<ValueIdx>& indptr,
           rmm::device_uvector<ValueIdx>& indices,
           rmm::device_uvector<value_t>& data,
           int c)
  {
    auto stream = raft::resource::get_cuda_stream(handle);

    size_t nnz = m * m;

    indices.resize(nnz, stream);
    data.resize(nnz, stream);

    pairwise_distances(handle, X, m, n, metric, indptr.data(), indices.data(), data.data());
  }
};

/**
 * Returns a CSR connectivities graph based on the given linkage distance.
 * @tparam ValueIdx
 * @tparam value_t
 * @tparam dist_type
 * @param[in] handle raft handle
 * @param[in] X dense data for which to construct connectivites
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metric to use
 * @param[out] indptr indptr array of connectivities graph
 * @param[out] indices column indices array of connectivities graph
 * @param[out] data distances array of connectivities graph
 * @param[out] c constant 'c' used for nearest neighbors-based distances
 *             which will guarantee k <= log(n) + c
 */
template <typename ValueIdx,
          typename value_t,
          Linkage dist_type>  // NOLINT(readability-identifier-naming)
void get_distance_graph(raft::resources const& handle,
                        const value_t* X,
                        ValueIdx m,
                        ValueIdx n,
                        cuvs::distance::DistanceType metric,
                        rmm::device_uvector<ValueIdx>& indptr,
                        rmm::device_uvector<ValueIdx>& indices,
                        rmm::device_uvector<value_t>& data,
                        int c)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  indptr.resize(m + 1, stream);

  distance_graph_impl<dist_type, ValueIdx, value_t> dist_graph;
  dist_graph.run(handle, X, m, n, metric, indptr, indices, data, c);
}

};  // namespace  cuvs::cluster::agglomerative::detail
