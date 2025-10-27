/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nn_descent.cuh"
#include <cuvs/neighbors/nn_descent.hpp>

namespace cuvs::neighbors::nn_descent {

#define CUVS_INST_NN_DESCENT_BUILD(T, IdxT)                                                   \
  auto build(raft::resources const& handle,                                                   \
             const cuvs::neighbors::nn_descent::index_params& params,                         \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,             \
             std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph) \
    -> cuvs::neighbors::nn_descent::index<IdxT>                                               \
  {                                                                                           \
    if (!graph.has_value()) {                                                                 \
      return cuvs::neighbors::nn_descent::build<T, IdxT>(handle, params, dataset);            \
    } else {                                                                                  \
      std::optional<raft::device_matrix_view<float, int64_t, raft::row_major>> distances =    \
        std::nullopt;                                                                         \
      cuvs::neighbors::nn_descent::index<IdxT> idx{                                           \
        handle, graph.value(), distances, params.metric};                                     \
      cuvs::neighbors::nn_descent::build<T, IdxT>(handle, params, dataset, idx);              \
      return idx;                                                                             \
    }                                                                                         \
  };                                                                                          \
                                                                                              \
  auto build(raft::resources const& handle,                                                   \
             const cuvs::neighbors::nn_descent::index_params& params,                         \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,               \
             std::optional<raft::host_matrix_view<uint32_t, int64_t, raft::row_major>> graph) \
    -> cuvs::neighbors::nn_descent::index<IdxT>                                               \
  {                                                                                           \
    if (!graph.has_value()) {                                                                 \
      return cuvs::neighbors::nn_descent::build<T, IdxT>(handle, params, dataset);            \
    } else {                                                                                  \
      std::optional<raft::device_matrix_view<float, int64_t, raft::row_major>> distances =    \
        std::nullopt;                                                                         \
      cuvs::neighbors::nn_descent::index<IdxT> idx{                                           \
        handle, graph.value(), distances, params.metric};                                     \
      cuvs::neighbors::nn_descent::build<T, IdxT>(handle, params, dataset, idx);              \
      return idx;                                                                             \
    }                                                                                         \
  };

CUVS_INST_NN_DESCENT_BUILD(int8_t, uint32_t);

#undef CUVS_INST_NN_DESCENT_BUILD

}  // namespace cuvs::neighbors::nn_descent
