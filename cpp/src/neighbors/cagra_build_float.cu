/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_BUILD(T, IdxT)                                                    \
  void build_knn_graph(raft::resources const& handle,                                     \
                       raft::host_matrix_view<const T, int64_t, raft::row_major> dataset, \
                       raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,  \
                       cuvs::neighbors::cagra::graph_build_params::ivf_pq_params params)  \
  {                                                                                       \
    cuvs::neighbors::cagra::build_knn_graph<T, IdxT>(handle, dataset, knn_graph, params); \
  }                                                                                       \
                                                                                          \
  auto build(raft::resources const& handle,                                               \
             const cuvs::neighbors::cagra::index_params& params,                          \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)         \
    -> cuvs::neighbors::cagra::index<T, IdxT>                                             \
  {                                                                                       \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);               \
  }                                                                                       \
                                                                                          \
  auto build(raft::resources const& handle,                                               \
             const cuvs::neighbors::cagra::index_params& params,                          \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)           \
    -> cuvs::neighbors::cagra::index<T, IdxT>                                             \
  {                                                                                       \
    return cuvs::neighbors::cagra::build<T, IdxT>(handle, params, dataset);               \
  }

RAFT_INST_CAGRA_BUILD(float, uint32_t);

template auto build(raft::resources const& res,
                    const cuvs::neighbors::cagra::index_params& params,
                    cuvs::neighbors::device_padded_dataset_view<float, int64_t> const& dataset)
  -> cuvs::neighbors::cagra::index<float, uint32_t>;
template auto build(raft::resources const& res,
                    const cuvs::neighbors::cagra::index_params& params,
                    cuvs::neighbors::device_padded_dataset<float, int64_t>&& dataset)
  -> cuvs::neighbors::cagra::index<float, uint32_t>;

#undef RAFT_INST_CAGRA_BUILD

}  // namespace cuvs::neighbors::cagra
