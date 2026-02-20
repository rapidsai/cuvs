/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/pca.cuh"

#include <cuvs/preprocessing/pca.hpp>

namespace cuvs::preprocessing::pca {

#define CUVS_INST_PCA_FIT(DataT, IndexT)                                        \
  void fit(raft::resources const& handle,                                       \
           params config,                                                       \
           raft::device_matrix_view<DataT, IndexT, raft::col_major> input,      \
           raft::device_matrix_view<DataT, IndexT, raft::col_major> components, \
           raft::device_vector_view<DataT, IndexT> explained_var,               \
           raft::device_vector_view<DataT, IndexT> explained_var_ratio,         \
           raft::device_vector_view<DataT, IndexT> singular_vals,               \
           raft::device_vector_view<DataT, IndexT> mu,                          \
           raft::device_scalar_view<DataT, IndexT> noise_vars,                  \
           bool flip_signs_based_on_U)                                          \
  {                                                                             \
    detail::fit(handle,                                                         \
                config,                                                         \
                input,                                                          \
                components,                                                     \
                explained_var,                                                  \
                explained_var_ratio,                                            \
                singular_vals,                                                  \
                mu,                                                             \
                noise_vars,                                                     \
                flip_signs_based_on_U);                                         \
  }

CUVS_INST_PCA_FIT(float, int64_t);
CUVS_INST_PCA_FIT(double, int64_t);
#undef CUVS_INST_PCA_FIT

#define CUVS_INST_PCA_FIT_TRANSFORM(DataT, IndexT)                                         \
  void fit_transform(raft::resources const& handle,                                        \
                     params config,                                                        \
                     raft::device_matrix_view<DataT, IndexT, raft::col_major> input,       \
                     raft::device_matrix_view<DataT, IndexT, raft::col_major> trans_input, \
                     raft::device_matrix_view<DataT, IndexT, raft::col_major> components,  \
                     raft::device_vector_view<DataT, IndexT> explained_var,                \
                     raft::device_vector_view<DataT, IndexT> explained_var_ratio,          \
                     raft::device_vector_view<DataT, IndexT> singular_vals,                \
                     raft::device_vector_view<DataT, IndexT> mu,                           \
                     raft::device_scalar_view<DataT, IndexT> noise_vars,                   \
                     bool flip_signs_based_on_U)                                           \
  {                                                                                        \
    detail::fit_transform(handle,                                                          \
                          config,                                                          \
                          input,                                                           \
                          trans_input,                                                     \
                          components,                                                      \
                          explained_var,                                                   \
                          explained_var_ratio,                                             \
                          singular_vals,                                                   \
                          mu,                                                              \
                          noise_vars,                                                      \
                          flip_signs_based_on_U);                                          \
  }

CUVS_INST_PCA_FIT_TRANSFORM(float, int64_t);
CUVS_INST_PCA_FIT_TRANSFORM(double, int64_t);
#undef CUVS_INST_PCA_FIT_TRANSFORM

#define CUVS_INST_PCA_TRANSFORM(DataT, IndexT)                                            \
  void transform(raft::resources const& handle,                                           \
                 params config,                                                           \
                 raft::device_matrix_view<DataT, IndexT, raft::col_major> input,          \
                 raft::device_matrix_view<DataT, IndexT, raft::col_major> components,     \
                 raft::device_vector_view<DataT, IndexT> singular_vals,                   \
                 raft::device_vector_view<DataT, IndexT> mu,                              \
                 raft::device_matrix_view<DataT, IndexT, raft::col_major> trans_input)    \
  {                                                                                       \
    detail::transform(handle, config, input, components, singular_vals, mu, trans_input); \
  }

CUVS_INST_PCA_TRANSFORM(float, int64_t);
CUVS_INST_PCA_TRANSFORM(double, int64_t);
#undef CUVS_INST_PCA_TRANSFORM

#define CUVS_INST_PCA_INVERSE_TRANSFORM(DataT, IndexT)                                             \
  void inverse_transform(raft::resources const& handle,                                            \
                         params config,                                                            \
                         raft::device_matrix_view<DataT, IndexT, raft::col_major> trans_input,     \
                         raft::device_matrix_view<DataT, IndexT, raft::col_major> components,      \
                         raft::device_vector_view<DataT, IndexT> singular_vals,                    \
                         raft::device_vector_view<DataT, IndexT> mu,                               \
                         raft::device_matrix_view<DataT, IndexT, raft::col_major> output)          \
  {                                                                                                \
    detail::inverse_transform(handle, config, trans_input, components, singular_vals, mu, output); \
  }

CUVS_INST_PCA_INVERSE_TRANSFORM(float, int64_t);
CUVS_INST_PCA_INVERSE_TRANSFORM(double, int64_t);
#undef CUVS_INST_PCA_INVERSE_TRANSFORM

}  // namespace cuvs::preprocessing::pca
