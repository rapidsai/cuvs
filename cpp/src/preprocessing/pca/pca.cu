/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/pca.cuh"

#include <cuvs/preprocessing/pca.hpp>

namespace cuvs::preprocessing::pca {

#define CUVS_INST_PCA_FIT(DataT)                                                 \
  void fit(raft::resources const& handle,                                        \
           params config,                                                        \
           raft::device_matrix_view<DataT, int64_t, raft::col_major> input,      \
           raft::device_matrix_view<DataT, int64_t, raft::col_major> components, \
           raft::device_vector_view<DataT, int64_t> explained_var,               \
           raft::device_vector_view<DataT, int64_t> explained_var_ratio,         \
           raft::device_vector_view<DataT, int64_t> singular_vals,               \
           raft::device_vector_view<DataT, int64_t> mu,                          \
           raft::device_scalar_view<DataT, int64_t> noise_vars,                  \
           bool flip_signs_based_on_U)                                           \
  {                                                                              \
    detail::fit(handle,                                                          \
                config,                                                          \
                input,                                                           \
                components,                                                      \
                explained_var,                                                   \
                explained_var_ratio,                                             \
                singular_vals,                                                   \
                mu,                                                              \
                noise_vars,                                                      \
                flip_signs_based_on_U);                                          \
  }

CUVS_INST_PCA_FIT(float);
CUVS_INST_PCA_FIT(double);
#undef CUVS_INST_PCA_FIT

#define CUVS_INST_PCA_FIT_TRANSFORM(DataT)                                                  \
  void fit_transform(raft::resources const& handle,                                         \
                     params config,                                                         \
                     raft::device_matrix_view<DataT, int64_t, raft::col_major> input,       \
                     raft::device_matrix_view<DataT, int64_t, raft::col_major> trans_input, \
                     raft::device_matrix_view<DataT, int64_t, raft::col_major> components,  \
                     raft::device_vector_view<DataT, int64_t> explained_var,                \
                     raft::device_vector_view<DataT, int64_t> explained_var_ratio,          \
                     raft::device_vector_view<DataT, int64_t> singular_vals,                \
                     raft::device_vector_view<DataT, int64_t> mu,                           \
                     raft::device_scalar_view<DataT, int64_t> noise_vars,                   \
                     bool flip_signs_based_on_U)                                            \
  {                                                                                         \
    detail::fit_transform(handle,                                                           \
                          config,                                                           \
                          input,                                                            \
                          trans_input,                                                      \
                          components,                                                       \
                          explained_var,                                                    \
                          explained_var_ratio,                                              \
                          singular_vals,                                                    \
                          mu,                                                               \
                          noise_vars,                                                       \
                          flip_signs_based_on_U);                                           \
  }

CUVS_INST_PCA_FIT_TRANSFORM(float);
CUVS_INST_PCA_FIT_TRANSFORM(double);
#undef CUVS_INST_PCA_FIT_TRANSFORM

#define CUVS_INST_PCA_TRANSFORM(DataT)                                                    \
  void transform(raft::resources const& handle,                                           \
                 params config,                                                           \
                 raft::device_matrix_view<DataT, int64_t, raft::col_major> input,         \
                 raft::device_matrix_view<DataT, int64_t, raft::col_major> components,    \
                 raft::device_vector_view<DataT, int64_t> singular_vals,                  \
                 raft::device_vector_view<DataT, int64_t> mu,                             \
                 raft::device_matrix_view<DataT, int64_t, raft::col_major> trans_input)   \
  {                                                                                       \
    detail::transform(handle, config, input, components, singular_vals, mu, trans_input); \
  }

CUVS_INST_PCA_TRANSFORM(float);
CUVS_INST_PCA_TRANSFORM(double);
#undef CUVS_INST_PCA_TRANSFORM

#define CUVS_INST_PCA_INVERSE_TRANSFORM(DataT)                                                     \
  void inverse_transform(raft::resources const& handle,                                            \
                         params config,                                                            \
                         raft::device_matrix_view<DataT, int64_t, raft::col_major> trans_input,    \
                         raft::device_matrix_view<DataT, int64_t, raft::col_major> components,     \
                         raft::device_vector_view<DataT, int64_t> singular_vals,                   \
                         raft::device_vector_view<DataT, int64_t> mu,                              \
                         raft::device_matrix_view<DataT, int64_t, raft::col_major> output)         \
  {                                                                                                \
    detail::inverse_transform(handle, config, trans_input, components, singular_vals, mu, output); \
  }

CUVS_INST_PCA_INVERSE_TRANSFORM(float);
CUVS_INST_PCA_INVERSE_TRANSFORM(double);
#undef CUVS_INST_PCA_INVERSE_TRANSFORM

}  // namespace cuvs::preprocessing::pca
