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

#pragma once

#include <raft/core/resources.hpp>
#include <raft/spectral/matrix_wrappers.hpp>

namespace cuvs {
namespace spectral {

/**
 * @brief Configuration parameters for eigen solver
 *
 * @tparam index_type_t Type for indices
 * @tparam value_type_t Type for values (float/double)
 * @tparam size_type_t Type for sizes (defaults to index_type_t)
 */
template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
struct eigen_solver_config_t {
  /** Number of eigenvectors to compute */
  size_type_t n_eigVecs;

  /** Maximum number of iterations */
  size_type_t maxIter;

  /** Number of iterations before restart */
  size_type_t restartIter;

  /** Tolerance for convergence */
  value_type_t tol;

  /** Whether to reorthogonalize eigenvectors */
  bool reorthogonalize = false;

  /**
   * Random seed for initialization
   * CAVEAT: this default value is now common to all instances of using seed in
   * Lanczos; was not the case before: there were places where a default seed = 123456
   * was used; this may trigger slightly different # solver iterations
   */
  unsigned long long seed = 1234567;
};

/**
 * @brief Lanczos solver for computing eigenvectors and eigenvalues
 *
 * This class provides methods to compute the smallest or largest eigenvectors
 * and eigenvalues of a sparse matrix using the Lanczos algorithm.
 *
 * @tparam index_type_t Type for indices
 * @tparam value_type_t Type for values (float/double)
 * @tparam size_type_t Type for sizes (defaults to index_type_t)
 */
template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
class lanczos_solver_t {
 public:
  /**
   * @brief Construct a new Lanczos solver
   *
   * @param config Configuration parameters for the solver
   */
  explicit lanczos_solver_t(
    eigen_solver_config_t<index_type_t, value_type_t, size_type_t> const& config);

  /**
   * @brief Compute the smallest eigenvectors and eigenvalues
   *
   * @param handle RAFT resource handle
   * @param A Sparse matrix for which to compute eigenvectors
   * @param eigVals Output buffer for eigenvalues (size: n_eigVecs)
   * @param eigVecs Output buffer for eigenvectors (size: n_rows * n_eigVecs)
   * @return Number of iterations performed
   */
  index_type_t solve_smallest_eigenvectors(
    raft::resources const& handle,
    raft::spectral::matrix::sparse_matrix_t<index_type_t, value_type_t, size_type_t> const& A,
    value_type_t* __restrict__ eigVals,
    value_type_t* __restrict__ eigVecs) const;

  /**
   * @brief Compute the largest eigenvectors and eigenvalues
   *
   * @param handle RAFT resource handle
   * @param A Sparse matrix for which to compute eigenvectors
   * @param eigVals Output buffer for eigenvalues (size: n_eigVecs)
   * @param eigVecs Output buffer for eigenvectors (size: n_rows * n_eigVecs)
   * @return Number of iterations performed
   */
  index_type_t solve_largest_eigenvectors(
    raft::resources const& handle,
    raft::spectral::matrix::sparse_matrix_t<index_type_t, value_type_t, size_type_t> const& A,
    value_type_t* __restrict__ eigVals,
    value_type_t* __restrict__ eigVecs) const;

  /**
   * @brief Get the configuration object
   *
   * @return const reference to the configuration
   */
  auto const& get_config() const;

 private:
  eigen_solver_config_t<index_type_t, value_type_t, size_type_t> config_;
};

}  // namespace spectral
}  // namespace cuvs
