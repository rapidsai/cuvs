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
#include "cuvs/neighbors/common.hpp"

namespace cuvs::neighbors::all_neighbors {

#define CUVS_INST_ALL_NEIGHBORS(T, IdxT)                                                        \
  all_neighbors::index<IdxT, T> build(const raft::resources& handle,                            \
                                      raft::host_matrix_view<const T, IdxT, row_major> dataset, \
                                      int64_t k,                                                \
                                      const index_params& params,                               \
                                      bool return_distances)                                    \
  {                                                                                             \
    return all_neighbors::detail::build<T, IdxT>(handle, dataset, k, params, return_distances); \
  }                                                                                             \
                                                                                                \
  void build(const raft::resources& handle,                                                     \
             raft::host_matrix_view<const T, IdxT, row_major> dataset,                          \
             const index_params& params,                                                        \
             all_neighbors::index<IdxT, T>& idx)                                                \
  {                                                                                             \
    return all_neighbors::detail::build<T, IdxT>(handle, dataset, params, idx);                 \
  }

CUVS_INST_ALL_NEIGHBORS(float, int64_t);

#undef CUVS_INST_ALL_NEIGHBORS

}  // namespace cuvs::neighbors::all_neighbors
