/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "eigen_solvers.cuh"

namespace cuvs::spectral {

#define CUVS_INST_LANCZOS_SOLVER(index_type, value_type, size_type)                      \
  template class lanczos_solver_t<index_type, value_type, size_type>;                    \
  template lanczos_solver_t<index_type, value_type, size_type>::lanczos_solver_t(        \
    eigen_solver_config_t<index_type, value_type, size_type> const& config);             \
  template index_type                                                                    \
  lanczos_solver_t<index_type, value_type, size_type>::solve_smallest_eigenvectors(      \
    raft::resources const& handle,                                                       \
    raft::spectral::matrix::sparse_matrix_t<index_type, value_type, size_type> const& A, \
    value_type* __restrict__ eigVals,                                                    \
    value_type* __restrict__ eigVecs) const;                                             \
  template index_type                                                                    \
  lanczos_solver_t<index_type, value_type, size_type>::solve_largest_eigenvectors(       \
    raft::resources const& handle,                                                       \
    raft::spectral::matrix::sparse_matrix_t<index_type, value_type, size_type> const& A, \
    value_type* __restrict__ eigVals,                                                    \
    value_type* __restrict__ eigVecs) const;                                             \
  template auto const& lanczos_solver_t<index_type, value_type, size_type>::get_config() const;

// Instantiate for common type combinations
CUVS_INST_LANCZOS_SOLVER(int, float, int);
CUVS_INST_LANCZOS_SOLVER(int, double, int);
CUVS_INST_LANCZOS_SOLVER(int64_t, float, int64_t);
CUVS_INST_LANCZOS_SOLVER(int64_t, double, int64_t);

#undef CUVS_INST_LANCZOS_SOLVER

}  // namespace cuvs::spectral
