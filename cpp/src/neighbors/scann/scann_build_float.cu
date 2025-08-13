/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "scann.cuh"
#include <cuvs/neighbors/scann.hpp>

namespace cuvs::neighbors::experimental::scann {

#define CUVS_INST_SCANN_BUILD(T, IdxT)                                                    \
  auto build(raft::resources const& handle,                                               \
             const cuvs::neighbors::experimental::scann::index_params& params,            \
             raft::device_matrix_view<const T, IdxT, raft::row_major> dataset)            \
    -> cuvs::neighbors::experimental::scann::index<T, IdxT>                               \
  {                                                                                       \
    return cuvs::neighbors::experimental::scann::build<T, IdxT>(handle, params, dataset); \
  }                                                                                       \
                                                                                          \
  auto build(raft::resources const& handle,                                               \
             const cuvs::neighbors::experimental::scann::index_params& params,            \
             raft::host_matrix_view<const T, IdxT, raft::row_major> dataset)              \
    -> cuvs::neighbors::experimental::scann::index<T, IdxT>                               \
  {                                                                                       \
    return cuvs::neighbors::experimental::scann::build<T, IdxT>(handle, params, dataset); \
  }

CUVS_INST_SCANN_BUILD(float, int64_t);

#undef CUVS_INST_SCANN_BUILD

}  // namespace cuvs::neighbors::experimental::scann
