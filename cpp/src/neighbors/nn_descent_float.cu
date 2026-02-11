/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "./detail/nn_descent_gnnd.hpp"
#include "./detail/reachability.cuh"
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
    };                                                                                        \
  }                                                                                           \
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
  };                                                                                          \
  template class detail::gnnd<const T, int>;                                                  \
                                                                                              \
  template void detail::gnnd<const T, int>::build<                                            \
    cuvs::neighbors::detail::reachability::reachability_post_process<int, T>>(                \
    const T* data,                                                                            \
    const int nrow,                                                                           \
    int* output_graph,                                                                        \
    bool return_distances,                                                                    \
    float* output_distances,                                                                  \
    cuvs::neighbors::detail::reachability::reachability_post_process<int, T> dist_epilogue);  \
  template void detail::gnnd<const T, int>::local_join<                                       \
    cuvs::neighbors::detail::reachability::reachability_post_process<int, T>>(                \
    cudaStream_t stream,                                                                      \
    cuvs::neighbors::detail::reachability::reachability_post_process<int, T> dist_epilogue);  \
                                                                                              \
  template void detail::gnnd<const T, int>::build<raft::identity_op>(                         \
    const T* data,                                                                            \
    const int nrow,                                                                           \
    int* output_graph,                                                                        \
    bool return_distances,                                                                    \
    float* output_distances,                                                                  \
    raft::identity_op dist_epilogue);                                                         \
  template void detail::gnnd<const T, int>::local_join<raft::identity_op>(                    \
    cudaStream_t stream, raft::identity_op dist_epilogue);

CUVS_INST_NN_DESCENT_BUILD(float, uint32_t);

#undef CUVS_INST_NN_DESCENT_BUILD

}  // namespace cuvs::neighbors::nn_descent
