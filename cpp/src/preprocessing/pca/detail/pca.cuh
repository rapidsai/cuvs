/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/preprocessing/pca.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/pca.cuh>

namespace cuvs::preprocessing::pca::detail {

/**
 * @brief Convert cuvs::preprocessing::pca::params to raft::linalg::paramsPCA.
 */
inline raft::linalg::paramsPCA to_raft_params(params config, std::size_t n_rows, std::size_t n_cols)
{
  raft::linalg::paramsPCA prms;
  prms.n_rows       = n_rows;
  prms.n_cols       = n_cols;
  prms.n_components = static_cast<std::size_t>(config.n_components);
  prms.algorithm    = static_cast<raft::linalg::solver>(static_cast<int>(config.algorithm));
  prms.tol          = config.tol;
  prms.n_iterations = static_cast<std::uint32_t>(config.n_iterations);
  prms.verbose      = config.verbose;
  prms.copy         = config.copy;
  prms.whiten       = config.whiten;
  return prms;
}

template <typename math_t>
void fit(raft::resources const& handle,
         params config,
         raft::device_matrix_view<math_t, int64_t, raft::col_major> input,
         raft::device_matrix_view<math_t, int64_t, raft::col_major> components,
         raft::device_vector_view<math_t, int64_t> explained_var,
         raft::device_vector_view<math_t, int64_t> explained_var_ratio,
         raft::device_vector_view<math_t, int64_t> singular_vals,
         raft::device_vector_view<math_t, int64_t> mu,
         raft::device_scalar_view<math_t, int64_t> noise_vars,
         bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(config, input.extent(0), input.extent(1));
  raft::linalg::pca_fit(handle,
                        raft_prms,
                        input,
                        components,
                        explained_var,
                        explained_var_ratio,
                        singular_vals,
                        mu,
                        noise_vars,
                        flip_signs_based_on_U);
}

template <typename math_t>
void fit_transform(raft::resources const& handle,
                   params config,
                   raft::device_matrix_view<math_t, int64_t, raft::col_major> input,
                   raft::device_matrix_view<math_t, int64_t, raft::col_major> trans_input,
                   raft::device_matrix_view<math_t, int64_t, raft::col_major> components,
                   raft::device_vector_view<math_t, int64_t> explained_var,
                   raft::device_vector_view<math_t, int64_t> explained_var_ratio,
                   raft::device_vector_view<math_t, int64_t> singular_vals,
                   raft::device_vector_view<math_t, int64_t> mu,
                   raft::device_scalar_view<math_t, int64_t> noise_vars,
                   bool flip_signs_based_on_U)
{
  auto raft_prms = to_raft_params(config, input.extent(0), input.extent(1));
  raft::linalg::pca_fit_transform(handle,
                                  raft_prms,
                                  input,
                                  trans_input,
                                  components,
                                  explained_var,
                                  explained_var_ratio,
                                  singular_vals,
                                  mu,
                                  noise_vars,
                                  flip_signs_based_on_U);
}

template <typename math_t>
void transform(raft::resources const& handle,
               params config,
               raft::device_matrix_view<math_t, int64_t, raft::col_major> input,
               raft::device_matrix_view<math_t, int64_t, raft::col_major> components,
               raft::device_vector_view<math_t, int64_t> singular_vals,
               raft::device_vector_view<math_t, int64_t> mu,
               raft::device_matrix_view<math_t, int64_t, raft::col_major> trans_input)
{
  auto raft_prms = to_raft_params(config, input.extent(0), input.extent(1));
  raft::linalg::pca_transform(handle, raft_prms, input, components, singular_vals, mu, trans_input);
}

template <typename math_t>
void inverse_transform(raft::resources const& handle,
                       params config,
                       raft::device_matrix_view<math_t, int64_t, raft::col_major> trans_input,
                       raft::device_matrix_view<math_t, int64_t, raft::col_major> components,
                       raft::device_vector_view<math_t, int64_t> singular_vals,
                       raft::device_vector_view<math_t, int64_t> mu,
                       raft::device_matrix_view<math_t, int64_t, raft::col_major> output)
{
  auto raft_prms = to_raft_params(config, output.extent(0), output.extent(1));
  raft::linalg::pca_inverse_transform(
    handle, raft_prms, trans_input, components, singular_vals, mu, output);
}

}  // namespace cuvs::preprocessing::pca::detail
