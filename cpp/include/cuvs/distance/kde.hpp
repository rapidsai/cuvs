/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <raft/core/resources.hpp>

namespace cuvs::distance {

/**
 * @brief Density kernel type for Kernel Density Estimation.
 *
 * These are the smoothing kernels used in KDE — distinct from the dot-product
 * kernels (RBF, Polynomial, etc.) in cuvs::distance::kernels used by SVMs.
 */
enum class DensityKernelType : int {
  Gaussian     = 0,
  Tophat       = 1,
  Epanechnikov = 2,
  Exponential  = 3,
  Linear       = 4,
  Cosine       = 5
};

/**
 * @brief Compute log-density estimates for query points using kernel density estimation.
 *
 * Fuses pairwise distance computation, kernel evaluation, logsumexp reduction,
 * and normalization into a single CUDA kernel pass. O(N+M) memory usage —
 * the full N×M pairwise distance matrix is never materialised.
 *
 * Supports 13 distance metrics (all expressible as per-feature accumulation),
 * 6 density kernel functions, float32 and float64, and both uniform and
 * weighted training sets.
 *
 * When the query count is small relative to the number of GPU SMs, the
 * training set is automatically split across a 2D grid (multi-pass mode) to
 * keep the GPU fully utilised. Partial logsumexp results are merged by a
 * reduction kernel.
 *
 * @tparam T  float or double
 *
 * @param[in]  handle      RAFT resources handle for stream management
 * @param[in]  query       Query points, row-major (n_query × n_features)
 * @param[in]  train       Training points, row-major (n_train × n_features)
 * @param[in]  weights     Per-training-point weights (n_train,), or nullptr for uniform
 * @param[out] output      Log-density estimates (n_query,)
 * @param[in]  n_query     Number of query points
 * @param[in]  n_train     Number of training points
 * @param[in]  n_features  Dimensionality of the data
 * @param[in]  bandwidth   Kernel bandwidth (must be > 0)
 * @param[in]  sum_weights Sum of sample weights (or n_train if uniform)
 * @param[in]  kernel      Density kernel function
 * @param[in]  metric      Distance metric
 * @param[in]  metric_arg  Metric parameter (e.g. p for Minkowski; ignored otherwise)
 */
template <typename T>
void kde_score_samples(raft::resources const& handle,
                       const T* query,
                       const T* train,
                       const T* weights,
                       T* output,
                       int n_query,
                       int n_train,
                       int n_features,
                       T bandwidth,
                       T sum_weights,
                       DensityKernelType kernel,
                       cuvs::distance::DistanceType metric,
                       T metric_arg);

extern template void kde_score_samples<float>(raft::resources const&,
                                              const float*,
                                              const float*,
                                              const float*,
                                              float*,
                                              int,
                                              int,
                                              int,
                                              float,
                                              float,
                                              DensityKernelType,
                                              cuvs::distance::DistanceType,
                                              float);

extern template void kde_score_samples<double>(raft::resources const&,
                                               const double*,
                                               const double*,
                                               const double*,
                                               double*,
                                               int,
                                               int,
                                               int,
                                               double,
                                               double,
                                               DensityKernelType,
                                               cuvs::distance::DistanceType,
                                               double);

}  // namespace cuvs::distance
