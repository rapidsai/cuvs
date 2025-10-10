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
#ifndef __EIGEN_SOLVERS_H
#define __EIGEN_SOLVERS_H

#pragma once

#include <cuvs/sparse/cluster/eigen_solvers.hpp>

#include <raft/sparse/solver/lanczos.cuh>
#include <raft/spectral/matrix_wrappers.hpp>

namespace cuvs {
namespace spectral {

template <typename index_type_t, typename value_type_t, typename size_type_t>
lanczos_solver_t<index_type_t, value_type_t, size_type_t>::lanczos_solver_t(
  eigen_solver_config_t<index_type_t, value_type_t, size_type_t> const& config)
  : config_(config)
{
}

template <typename index_type_t, typename value_type_t, typename size_type_t>
index_type_t lanczos_solver_t<index_type_t, value_type_t, size_type_t>::solve_smallest_eigenvectors(
  raft::resources const& handle,
  raft::spectral::matrix::sparse_matrix_t<index_type_t, value_type_t, size_type_t> const& A,
  value_type_t* __restrict__ eigVals,
  value_type_t* __restrict__ eigVecs) const
{
  RAFT_EXPECTS(eigVals != nullptr, "Null eigVals buffer.");
  RAFT_EXPECTS(eigVecs != nullptr, "Null eigVecs buffer.");
  index_type_t iters{};
  raft::sparse::solver::computeSmallestEigenvectors(handle,
                                                    A,
                                                    config_.n_eigVecs,
                                                    config_.maxIter,
                                                    config_.restartIter,
                                                    config_.tol,
                                                    config_.reorthogonalize,
                                                    iters,
                                                    eigVals,
                                                    eigVecs,
                                                    config_.seed);
  return iters;
}

template <typename index_type_t, typename value_type_t, typename size_type_t>
index_type_t lanczos_solver_t<index_type_t, value_type_t, size_type_t>::solve_largest_eigenvectors(
  raft::resources const& handle,
  raft::spectral::matrix::sparse_matrix_t<index_type_t, value_type_t, size_type_t> const& A,
  value_type_t* __restrict__ eigVals,
  value_type_t* __restrict__ eigVecs) const
{
  RAFT_EXPECTS(eigVals != nullptr, "Null eigVals buffer.");
  RAFT_EXPECTS(eigVecs != nullptr, "Null eigVecs buffer.");
  index_type_t iters{};
  raft::sparse::solver::computeLargestEigenvectors(handle,
                                                   A,
                                                   config_.n_eigVecs,
                                                   config_.maxIter,
                                                   config_.restartIter,
                                                   config_.tol,
                                                   config_.reorthogonalize,
                                                   iters,
                                                   eigVals,
                                                   eigVecs,
                                                   config_.seed);
  return iters;
}

template <typename index_type_t, typename value_type_t, typename size_type_t>
auto const& lanczos_solver_t<index_type_t, value_type_t, size_type_t>::get_config() const
{
  return config_;
}

}  // namespace spectral
}  // namespace cuvs

#endif
