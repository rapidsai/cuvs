/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cuvs/neighbors/ivf_pq.cuh>
#include <raft_runtime/neighbors/ivf_pq.hpp>

namespace cuvs::runtime::neighbors::ivf_pq {

#define RAFT_INST_BUILD_EXTEND(T, IdxT)                                                     \
  cuvs::neighbors::ivf_pq::index<IdxT> build(                                               \
    raft::resources const& handle,                                                          \
    const cuvs::neighbors::ivf_pq::index_params& params,                                    \
    raft::device_matrix_view<const T, IdxT, row_major> dataset)                             \
  {                                                                                         \
    return cuvs::neighbors::ivf_pq::build<T, IdxT>(handle, params, dataset);                \
  }                                                                                         \
  void build(raft::resources const& handle,                                                 \
             const cuvs::neighbors::ivf_pq::index_params& params,                           \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,                    \
             cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                     \
  {                                                                                         \
    *idx = cuvs::neighbors::ivf_pq::build<T, IdxT>(handle, params, dataset);                \
  }                                                                                         \
  cuvs::neighbors::ivf_pq::index<IdxT> extend(                                              \
    raft::resources const& handle,                                                          \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                         \
    std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,                  \
    const cuvs::neighbors::ivf_pq::index<IdxT>& idx)                                        \
  {                                                                                         \
    return cuvs::neighbors::ivf_pq::extend<T, IdxT>(handle, new_vectors, new_indices, idx); \
  }                                                                                         \
  void extend(raft::resources const& handle,                                                \
              raft::device_matrix_view<const T, IdxT, row_major> new_vectors,               \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,        \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                    \
  {                                                                                         \
    cuvs::neighbors::ivf_pq::extend<T, IdxT>(handle, new_vectors, new_indices, idx);        \
  }

RAFT_INST_BUILD_EXTEND(float, int64_t);
RAFT_INST_BUILD_EXTEND(int8_t, int64_t);
RAFT_INST_BUILD_EXTEND(uint8_t, int64_t);

#undef RAFT_INST_BUILD_EXTEND

}  // namespace cuvs::runtime::neighbors::ivf_pq
