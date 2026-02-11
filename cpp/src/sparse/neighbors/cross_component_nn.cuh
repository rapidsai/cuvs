/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/cross_component_nn.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>

namespace cuvs::sparse::neighbors {

template <typename ValueIdx, typename ValueT>
using fix_connectivities_red_op = detail::fix_connectivities_red_op<ValueIdx, ValueT>;

template <typename ValueIdx, typename ValueT>
using mutual_reachability_fix_connectivities_red_op =
  detail::mutual_reachability_fix_connectivities_red_op<ValueIdx, ValueT>;

/**
 * Gets the number of unique components from array of
 * colors or labels. This does not assume the components are
 * drawn from a monotonically increasing set.
 * @tparam ValueIdx
 * @param[in] colors array of components
 * @param[in] n_rows size of components array
 * @param[in] stream cuda stream for which to order cuda operations
 * @return total number of components
 */
template <typename ValueIdx>
auto get_n_components(ValueIdx* colors, size_t n_rows, cudaStream_t stream) -> ValueIdx
{
  return detail::get_n_components(colors, n_rows, stream);
}

/**
 * Connects the components of an otherwise unconnected knn graph
 * by computing a 1-nn to neighboring components of each data point
 * (e.g. component(nn) != component(self)) and reducing the results to
 * include the set of smallest destination components for each source
 * component. The result will not necessarily contain
 * n_components^2 - n_components number of elements because many components
 * will likely not be contained in the neighborhoods of 1-nns.
 * @tparam ValueIdx
 * @tparam ValueT
 * @param[in] handle raft handle
 * @param[out] out output edge list containing nearest cross-component
 *             edges.
 * @param[in] X original (row-major) dense matrix for which knn graph should be constructed.
 * @param[in] orig_colors array containing component number for each row of X
 * @param[in] n_rows number of rows in X
 * @param[in] n_cols number of cols in X
 * @param[in] reduction_op reduction operation for computing nearest neighbors. The reduction
 * operation must have `gather` and `scatter` functions defined
 * @param[in] row_batch_size the batch size for computing nearest neighbors. This parameter controls
 * the number of samples for which the nearest neighbors are computed at once. Therefore, it affects
 * the memory consumption mainly by reducing the size of the adjacency matrix for masked nearest
 * neighbors computation
 * @param[in] col_batch_size the input data is sorted and 'unsorted' based on color. An additional
 * scratch space buffer of shape (n_rows, col_batch_size) is created for this. Usually, this
 * parameter affects the memory consumption more drastically than the row_batch_size with a marginal
 * increase in compute time as the col_batch_size is reduced
 * @param[in] metric distance metric
 */
template <typename ValueIdx, typename ValueT, typename RedOp>
void cross_component_nn(
  raft::resources const& handle,
  raft::sparse::COO<ValueT, ValueIdx>& out,
  const ValueT* X,
  const ValueIdx* orig_colors,
  ValueIdx n_rows,
  ValueIdx n_cols,
  RedOp reduction_op,
  ValueIdx row_batch_size             = 0,
  ValueIdx col_batch_size             = 0,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded)
{
  detail::cross_component_nn(handle,
                             out,
                             X,
                             orig_colors,
                             n_rows,
                             n_cols,
                             reduction_op,
                             row_batch_size,
                             col_batch_size,
                             metric);
}

};  // end namespace cuvs::sparse::neighbors
