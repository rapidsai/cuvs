/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/cagra.cuh>
#include <cuvs/neighbors/cagra_types.hpp>
#include <raft_runtime/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define CUVS_INST_CAGRA_BUILD(T, IdxT)                                                   \
  auto build(raft::resources const& handle,                                              \
             const cuvs::neighbors::cagra::index_params& params,                         \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)        \
    ->cuvs::neighbors::cagra::index<T, IdxT>                                             \
  {                                                                                      \
    return cuvs::neighbors::cagra::index<T, IdxT>(                                       \
      std::move(raft::runtime::neighbors::cagra::build(handle, params, dataset)));  \
  }                                                                                      \
                                                                                         \
  auto build(raft::resources const& handle,                                              \
             const cuvs::neighbors::cagra::index_params& params,                         \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)          \
    ->cuvs::neighbors::cagra::index<T, IdxT>                                             \
  {                                                                                      \
    return cuvs::neighbors::cagra::index<T, IdxT>(                                       \
      std::move(raft::runtime::neighbors::cagra::build(handle, params, dataset)));  \
  }                                                                                      \
                                                                                         \
  void build_device(raft::resources const& handle,                                       \
                    const cuvs::neighbors::cagra::index_params& params,                  \
                    raft::device_matrix_view<const T, int64_t, raft::row_major> dataset, \
                    cuvs::neighbors::cagra::index<T, IdxT>& idx)                         \
  {                                                                                      \
    raft::runtime::neighbors::cagra::build_device(                                       \
      handle, params, dataset, *idx.get_raft_index());                              \
  }                                                                                      \
                                                                                         \
  void build_host(raft::resources const& handle,                                         \
                  const cuvs::neighbors::cagra::index_params& params,                    \
                  raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,     \
                  cuvs::neighbors::cagra::index<T, IdxT>& idx)                           \
  {                                                                                      \
    raft::runtime::neighbors::cagra::build_host(                                         \
      handle, params, dataset, *idx.get_raft_index());                              \
  }

CUVS_INST_CAGRA_BUILD(float, uint32_t);
CUVS_INST_CAGRA_BUILD(int8_t, uint32_t);
CUVS_INST_CAGRA_BUILD(uint8_t, uint32_t);

#undef CUVS_INST_CAGRA_BUILD

#define CUVS_INST_CAGRA_OPTIMIZE(IdxT)                                                     \
  void optimize_device(raft::resources const& handle,                                      \
                       raft::device_matrix_view<IdxT, int64_t, raft::row_major> knn_graph, \
                       raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph)   \
  {                                                                                        \
    raft::runtime::neighbors::cagra::optimize_device(handle, knn_graph, new_graph);        \
  }                                                                                        \
  void optimize_host(raft::resources const& handle,                                        \
                     raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,     \
                     raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph)     \
  {                                                                                        \
    raft::runtime::neighbors::cagra::optimize_host(handle, knn_graph, new_graph);          \
  }

CUVS_INST_CAGRA_OPTIMIZE(uint32_t);

#undef CUVS_INST_CAGRA_OPTIMIZE

}  // namespace cuvs::neighbors::cagra
