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

#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft_runtime/neighbors/ivf_pq.hpp>

namespace cuvs::neighbors::ivf_pq {

#define CUVS_INST_IVF_PQ_SERIALIZE(IdxT)                                                      \
  void serialize(raft::resources const& handle,                                               \
                 std::string& filename,                                                       \
                 const cuvs::neighbors::ivf_pq::index<IdxT>& index)                           \
  {                                                                                           \
    raft::runtime::neighbors::ivf_pq::serialize(handle, filename, *index.get_raft_index());   \
  }                                                                                           \
  void deserialize(raft::resources const& handle,                                             \
                   const std::string& filename,                                               \
                   cuvs::neighbors::ivf_pq::index<IdxT>* index)                               \
  {                                                                                           \
    raft::runtime::neighbors::ivf_pq::deserialize(handle, filename, index->get_raft_index()); \
  }

CUVS_INST_IVF_PQ_SERIALIZE(int64_t);

#undef CUVS_INST_IVF_PQ_SERIALIZE

}  // namespace cuvs::neighbors::ivf_pq