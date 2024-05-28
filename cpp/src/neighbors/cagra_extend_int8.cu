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

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_EXTEND(T, IdxT)                                                       \
  void add_graph_nodes(                                                                       \
    raft::resources const& handle,                                                            \
    raft::device_mdspan<const T, raft::matrix_extent<int64_t>, raft::layout_stride>           \
      updated_dataset_view,                                                                   \
    const cuvs::neighbors::cagra::index<T, IdxT>& idx,                                        \
    raft::host_matrix_view<IdxT, std::int64_t> updated_graph_view,                            \
    const cagra::extend_params& params)                                                       \
  {                                                                                           \
    cuvs::neighbors::cagra::add_graph_nodes<T, IdxT>(                                         \
      handle, updated_dataset_view, idx, updated_graph_view, params);                         \
  }                                                                                           \
                                                                                              \
  void add_graph_nodes(                                                                       \
    raft::resources const& handle,                                                            \
    raft::host_mdspan<const T, raft::matrix_extent<int64_t>, raft::layout_stride>             \
      updated_dataset_view,                                                                   \
    const cuvs::neighbors::cagra::index<T, IdxT>& idx,                                        \
    raft::host_matrix_view<IdxT, std::int64_t> updated_graph_view,                            \
    const cagra::extend_params& params)                                                       \
  {                                                                                           \
    cuvs::neighbors::cagra::add_graph_nodes<T, IdxT>(                                         \
      handle, updated_dataset_view, idx, updated_graph_view, params);                         \
  }                                                                                           \
                                                                                              \
  void extend(raft::resources const& handle,                                                  \
              raft::device_matrix_view<const T, int64_t, raft::row_major> additional_dataset, \
              cuvs::neighbors::cagra::index<T, IdxT>& idx,                                    \
              const cagra::extend_params& params)                                             \
  {                                                                                           \
    cuvs::neighbors::cagra::extend<T, IdxT>(handle, additional_dataset, idx, params);         \
  }                                                                                           \
                                                                                              \
  void extend(raft::resources const& handle,                                                  \
              raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset,   \
              cuvs::neighbors::cagra::index<T, IdxT>& idx,                                    \
              const cagra::extend_params& params)                                             \
  {                                                                                           \
    cuvs::neighbors::cagra::extend<T, IdxT>(handle, additional_dataset, idx, params);         \
  }

RAFT_INST_CAGRA_EXTEND(int8_t, uint32_t);

#undef RAFT_INST_CAGRA_EXTEND

}  // namespace cuvs::neighbors::cagra
