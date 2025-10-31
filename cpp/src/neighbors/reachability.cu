/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/reachability.hpp>

#include "./detail/reachability.cuh"

namespace cuvs::neighbors::reachability {

void mutual_reachability_graph(const raft::resources& handle,
                               raft::device_matrix_view<const float, int, raft::row_major> X,
                               int min_samples,
                               raft::device_vector_view<int> indptr,
                               raft::device_vector_view<float> core_dists,
                               raft::sparse::COO<float, int, size_t>& out,
                               cuvs::distance::DistanceType metric,
                               float alpha)
{
  RAFT_EXPECTS(core_dists.extent(0) == static_cast<size_t>(X.extent(0)),
               "core_dists doesn't have expected size");
  RAFT_EXPECTS(indptr.extent(0) == static_cast<size_t>(X.extent(0) + 1),
               "indptr doesn't have expected size");

  cuvs::neighbors::detail::reachability::mutual_reachability_graph<int, float, size_t>(
    handle,
    X.data_handle(),
    X.extent(0),
    X.extent(1),
    metric,
    min_samples,
    alpha,
    indptr.data_handle(),
    core_dists.data_handle(),
    out);
}
}  // namespace cuvs::neighbors::reachability
