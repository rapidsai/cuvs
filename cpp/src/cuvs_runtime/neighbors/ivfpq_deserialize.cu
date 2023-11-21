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
#include <cuvs/neighbors/ivf_pq_serialize.cuh>

#include <raft_runtime/neighbors/ivf_pq.hpp>

namespace cuvs::runtime::neighbors::ivf_pq {

void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  *index = cuvs::neighbors::ivf_pq::deserialize<int64_t>(handle, filename);
};
}  // namespace cuvs::runtime::neighbors::ivf_pq
