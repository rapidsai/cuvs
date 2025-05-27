/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "all_neighbors.cuh"

namespace cuvs::neighbors::all_neighbors {

#define CUVS_INST_ALL_NEIGHBORS(T, IdxT)                                                       \
  void build(const raft::resources& handle,                                                    \
             const all_neighbors_params& params,                                               \
             raft::host_matrix_view<const T, IdxT, row_major> dataset,                         \
             raft::device_matrix_view<IdxT, IdxT, row_major> indices,                          \
             std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances)            \
  {                                                                                            \
    return all_neighbors::detail::build<T, IdxT>(handle, params, dataset, indices, distances); \
  }                                                                                            \
                                                                                               \
  void build(const raft::resources& handle,                                                    \
             const all_neighbors_params& params,                                               \
             raft::device_matrix_view<const T, IdxT, row_major> dataset,                       \
             raft::device_matrix_view<IdxT, IdxT, row_major> indices,                          \
             std::optional<raft::device_matrix_view<T, IdxT, row_major>> distances)            \
  {                                                                                            \
    return all_neighbors::detail::build<T, IdxT>(handle, params, dataset, indices, distances); \
  }

CUVS_INST_ALL_NEIGHBORS(float, int64_t);

#undef CUVS_INST_ALL_NEIGHBORS

}  // namespace cuvs::neighbors::all_neighbors
