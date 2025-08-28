/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "detail/epsilon_neighborhood.cuh"
#include <cuvs/neighbors/epsilon_neighborhood.hpp>
#include <raft/util/cudart_utils.hpp>

namespace cuvs::neighbors::epsilon_neighborhood {

template <typename value_t, typename idx_t>
void epsUnexpL2SqNeighborhood(bool* adj,
                              idx_t* vd,
                              const value_t* x,
                              const value_t* y,
                              idx_t m,
                              idx_t n,
                              idx_t k,
                              value_t eps,
                              cudaStream_t stream)
{
  detail::epsUnexpL2SqNeighborhood<value_t, idx_t>(adj, vd, x, y, m, n, k, eps, stream);
}

template <typename value_t, typename idx_t, typename matrix_idx_t>
void compute(raft::resources const& handle,
             raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> x,
             raft::device_matrix_view<const value_t, matrix_idx_t, raft::row_major> y,
             raft::device_matrix_view<bool, matrix_idx_t, raft::row_major> adj,
             raft::device_vector_view<idx_t, matrix_idx_t> vd,
             value_t eps,
             cuvs::distance::DistanceType metric)
{
  // Currently only L2Unexpanded metric is supported
  RAFT_EXPECTS(metric == cuvs::distance::DistanceType::L2Unexpanded,
               "Currently only L2Unexpanded distance metric is supported. "
               "Other metrics will be supported in future versions.");

  epsUnexpL2SqNeighborhood<value_t, idx_t>(adj.data_handle(),
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
