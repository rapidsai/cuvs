/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/all_neighbors.hpp>

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cuda_dev_essentials.cuh>

#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace cuvs::neighbors::detail {

template <typename ValueIdx>
auto build_k(ValueIdx n_samples, int c) -> ValueIdx
{
  // from "kNN-MST-Agglomerative: A fast & scalable graph-based data clustering
  // approach on GPU"
  return std::min(
    n_samples,
    std::max(static_cast<ValueIdx>(2), static_cast<ValueIdx>(floor(raft::log2(n_samples))) + c));
}
/**
 * Constructs a (symmetrized) knn graph edge list from
 * dense input vectors.
 *
 * Note: The resulting KNN graph is not guaranteed to be connected.
 *
 * @tparam ValueIdx
 * @tparam ValueT
 * @tparam NnzT
 * @param[in] res raft res
 * @param[in] X dense matrix of input data samples and observations (size m * n)
 * @param[in] metric distance metric to use when constructing neighborhoods
 * @param[out] out output edge list
 * @param[in] c a constant used when constructing linkage from knn graph. Allows the indirect
 control of k. The algorithm will set `k = log(n) + c`
 */
template <typename ValueIdx = int,
          typename ValueT   = float,
          typename NnzT     = size_t>  // NOLINT(readability-identifier-naming)
void knn_graph(raft::resources const& res,
               raft::device_matrix_view<const ValueT, ValueIdx> X,
               cuvs::distance::DistanceType metric,
               raft::sparse::COO<ValueT, ValueIdx, NnzT>& out,
               int c = 15)
{
  size_t m = X.extent(0);
  size_t n = X.extent(1);
  size_t k = build_k(m, c);

  auto stream = raft::resource::get_cuda_stream(res);

  NnzT nnz = m * k;

  rmm::device_uvector<ValueIdx> rows(nnz, stream);
  rmm::device_uvector<ValueIdx> indices(nnz, stream);
  rmm::device_uvector<ValueT> data(nnz, stream);

  auto rows_view = raft::make_device_vector_view<ValueIdx, NnzT>(rows.data(), nnz);

  raft::linalg::map_offset(
    res, rows_view, [k] __device__(NnzT i) -> ValueIdx { return ValueIdx(i / k); });

  cuvs::neighbors::all_neighbors::all_neighbors_params params;
  params.metric = metric;

  cuvs::neighbors::graph_build_params::brute_force_params bf_params;
  bf_params.build_params.metric = metric;
  params.graph_build_params     = bf_params;

  params.n_clusters     = 1;
  params.overlap_factor = 1;

  rmm::device_uvector<int64_t> indices_64(nnz, stream);
  auto indices_64_view = raft::make_device_matrix_view<int64_t, int64_t>(indices_64.data(), m, k);
  auto distances_view  = raft::make_device_matrix_view<ValueT, int64_t>(data.data(), m, k);

  cuvs::neighbors::all_neighbors::build(
    res,
    params,
    raft::make_device_matrix_view<const ValueT, int64_t>(X.data_handle(), m, n),
    indices_64_view,
    distances_view);

  raft::linalg::unary_op(res,
                         raft::make_const_mdspan(indices_64_view),
                         raft::make_device_vector_view<ValueIdx, NnzT>(indices.data(), nnz),
                         raft::cast_op<ValueIdx>{});

  raft::sparse::linalg::symmetrize(res,
                                   rows.data(),
                                   indices.data(),
                                   data.data(),
                                   static_cast<ValueIdx>(m),
                                   static_cast<ValueIdx>(k),
                                   nnz,
                                   out);
}
};  // namespace cuvs::neighbors::detail
