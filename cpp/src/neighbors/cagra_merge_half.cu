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

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

#define RAFT_INST_CAGRA_MERGE(T, IdxT)                                      \
  auto merge(raft::resources const& handle,                                 \
             const cuvs::neighbors::cagra::merge_params& params,            \
             std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices) \
    -> cuvs::neighbors::cagra::index<T, IdxT>                               \
  {                                                                         \
    return cuvs::neighbors::cagra::merge<T, IdxT>(handle, params, indices); \
  }

RAFT_INST_CAGRA_MERGE(half, uint32_t);

#undef RAFT_INST_CAGRA_MERGE

}  // namespace cuvs::neighbors::cagra
