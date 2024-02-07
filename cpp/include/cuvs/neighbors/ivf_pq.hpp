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

#pragma once

#include <raft_runtime/neighbors/ivf_pq.hpp>


namespace cuvs::neighbors::ivf_pq {

using index_params = raft::neighbors::ivf_pq::index_params;
using search_params = raft::neighbors::ivf_pq::search_params;

template <typename IdxT>
using index = raft::neighbors::ivf_pq::index<IdxT>;

#define CUVS_IVF_PQ(T, IdxT)                                                                    \
  auto build(raft::resources const& handle,                                                     \
             const cuvs::neighbors::ivf_pq::index_params& params,                               \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset);                 \
                                                                                                \
  void build(raft::resources const& handle,                                                     \
             const cuvs::neighbors::ivf_pq::index_params& params,                               \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset,                  \
             cuvs::neighbors::ivf_pq::index<IdxT>* idx);                                        \
                                                                                                \
  auto extend(raft::resources const& handle,                                                    \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,             \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,            \
              const cuvs::neighbors::ivf_pq::index<IdxT>& orig_index);                          \
                                                                                                \
  void extend(raft::resources const& handle,                                                    \
              raft::device_matrix_view<const T, IdxT, raft::row_major> new_vectors,             \
              std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,            \
              cuvs::neighbors::ivf_pq::index<IdxT>* idx);                                       \
                                                                                                \
  void search(raft::resources const& handle,                                                    \
              const cuvs::neighbors::ivf_pq::search_params& params,                             \
              cuvs::neighbors::ivf_pq::index<IdxT>& index,                                      \
              raft::device_matrix_view<const T, IdxT, raft::row_major> queries,                 \
              raft::device_matrix_view<IdxT, IdxT, raft::row_major> neighbors,                  \
              raft::device_matrix_view<float, IdxT, raft::row_major> distances);                \
                                                                                                \
  void serialize(raft::resources const& handle,                                                 \
                 std::string& filename,                                                         \
                 const cuvs::neighbors::ivf_pq::index<IdxT>& index);                            \
                                                                                                \
  void deserialize(raft::resources const& handle,                                               \
                   const std::string& filename,                                                 \
                   cuvs::neighbors::ivf_pq::index<IdxT>* index);

CUVS_IVF_PQ(float, uint64_t);
CUVS_IVF_PQ(int8_t, uint64_t);
CUVS_IVF_PQ(uint8_t, uint64_t);

#undef CUVS_IVF_PQ

}  // namespace cuvs::neighbors::ivf_pq
