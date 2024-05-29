/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

/*
 * NOTE: this file is used by generate_ivf_pq.py
 *
 */

#include <cuvs/neighbors/ivf_pq.hpp>

#include "../ivf_pq_build.cuh"

namespace cuvs::neighbors::ivf_pq {

#define CUVS_INST_IVF_PQ_BUILD_EXTEND(T, IdxT)                                                    \
  auto build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset)                 \
    ->cuvs::neighbors::ivf_pq::index<IdxT>                                                        \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset);                       \
  }                                                                                               \
                                                                                                  \
  void build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,                 \
             cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                           \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset, idx);                         \
  }                                                                                               \
                                                                                                  \
  auto build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset)                   \
    ->cuvs::neighbors::ivf_pq::index<IdxT>                                                        \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset);                       \
  }                                                                                               \
                                                                                                  \
  void build(raft::resources const& handle,                                                       \
             const cuvs::neighbors::ivf_pq::index_params& params,                                 \
             raft::host_matrix_view<const T, int64_t, raft::row_major> dataset,                   \
             cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                           \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::build(handle, params, dataset, idx);                         \
  }                                                                                               \
  auto extend(                                                                                    \
    raft::resources const& handle,                                                                \
    raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,                      \
    std::optional<raft::device_vector_view<const IdxT, int64_t, raft::row_major>> new_indices,    \
    const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index)                                       \
    ->cuvs::neighbors::ivf_pq::index<IdxT>                                                        \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, orig_index); \
  }                                                                                               \
  void extend(raft::resources const& handle,                                                      \
              raft::device_matrix_view<const T, int64_t, raft::row_major> new_vectors,            \
              std::optional<raft::device_vector_view<const IdxT, int64_t>> new_indices,           \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                          \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, idx);               \
  }                                                                                               \
  auto extend(raft::resources const& handle,                                                      \
              raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,              \
              std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices,             \
              const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index)                             \
    ->cuvs::neighbors::ivf_pq::index<IdxT>                                                        \
  {                                                                                               \
    return cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, orig_index); \
  }                                                                                               \
                                                                                                  \
  void extend(raft::resources const& handle,                                                      \
              raft::host_matrix_view<const T, int64_t, raft::row_major> new_vectors,              \
              std::optional<raft::host_vector_view<const IdxT, int64_t>> new_indices,             \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx)                                          \
  {                                                                                               \
    cuvs::neighbors::ivf_pq::detail::extend(handle, new_vectors, new_indices, idx);               \
  }

}  // namespace cuvs::neighbors::ivf_pq
