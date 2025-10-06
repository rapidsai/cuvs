/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
  }                                                                                       \
                                                                                          \
  auto build_ace(raft::resources const& handle,                                           \
                 const cuvs::neighbors::cagra::index_params& params,                      \
                 raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,       \
                 size_t num_clusters) -> cuvs::neighbors::cagra::index<T, IdxT>           \
  {                                                                                       \
    return cuvs::neighbors::cagra::detail::build_ace<T, IdxT>(                            \
      handle, params, dataset, num_clusters);                                             \
  }

RAFT_INST_CAGRA_BUILD(int8_t, uint32_t);

#undef RAFT_INST_CAGRA_BUILD

}  // namespace cuvs::neighbors::cagra
