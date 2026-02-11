/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/epsilon_neighborhood.cuh"
#include <cuvs/neighbors/epsilon_neighborhood.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::epsilon_neighborhood {

template <typename ValueT, typename IdxT>     // NOLINT(readability-identifier-naming)
void eps_unexp_l2_sq_neighborhood(bool* adj,  // NOLINT(readability-identifier-naming)
                                  IdxT* vd,
                                  const ValueT* x,
                                  const ValueT* y,
                                  IdxT m,
                                  IdxT n,
                                  IdxT k,
                                  ValueT eps,
                                  cudaStream_t stream)
{
  detail::eps_unexp_l2_sq_neighborhood<ValueT, IdxT>(adj, vd, x, y, m, n, k, eps, stream);
}

template <typename ValueT,  // NOLINT(readability-identifier-naming)
          typename IdxT,
          typename matrix_idx_t>  // NOLINT(readability-identifier-naming)
void compute(raft::resources const& handle,
             raft::device_matrix_view<const ValueT, matrix_idx_t, raft::row_major> x,
             raft::device_matrix_view<const ValueT, matrix_idx_t, raft::row_major> y,
             raft::device_matrix_view<bool, matrix_idx_t, raft::row_major> adj,
             raft::device_vector_view<IdxT, matrix_idx_t> vd,
             ValueT eps,
             cuvs::distance::DistanceType metric)
{
  // Currently only L2Unexpanded metric is supported
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Unexpanded,
               "Currently only L2Unexpanded distance metric is supported. "
               "Other metrics will be supported in future versions.");

  eps_unexp_l2_sq_neighborhood<ValueT, IdxT>(adj.data_handle(),
                                             vd.data_handle(),
                                             x.data_handle(),
                                             y.data_handle(),
                                             x.extent(0),
                                             y.extent(0),
                                             x.extent(1),
                                             eps,
                                             raft::resource::get_cuda_stream(handle));
}

// Explicit template instantiations
template void compute<float, int64_t, int64_t>(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> x,
  raft::device_matrix_view<const float, int64_t, raft::row_major> y,
  raft::device_matrix_view<bool, int64_t, raft::row_major> adj,
  raft::device_vector_view<int64_t, int64_t> vd,
  float eps,
  cuvs::distance::DistanceType metric);

template void compute<float, int, int64_t>(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> x,
  raft::device_matrix_view<const float, int64_t, raft::row_major> y,
  raft::device_matrix_view<bool, int64_t, raft::row_major> adj,
  raft::device_vector_view<int, int64_t> vd,
  float eps,
  cuvs::distance::DistanceType metric);

template void compute<double, int, int64_t>(
  raft::resources const& handle,
  raft::device_matrix_view<const double, int64_t, raft::row_major> x,
  raft::device_matrix_view<const double, int64_t, raft::row_major> y,
  raft::device_matrix_view<bool, int64_t, raft::row_major> adj,
  raft::device_vector_view<int, int64_t> vd,
  double eps,
  cuvs::distance::DistanceType metric);

template void compute<double, int64_t, int64_t>(
  raft::resources const& handle,
  raft::device_matrix_view<const double, int64_t, raft::row_major> x,
  raft::device_matrix_view<const double, int64_t, raft::row_major> y,
  raft::device_matrix_view<bool, int64_t, raft::row_major> adj,
  raft::device_vector_view<int64_t, int64_t> vd,
  double eps,
  cuvs::distance::DistanceType metric);

}  // namespace cuvs::neighbors::epsilon_neighborhood
