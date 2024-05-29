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

#include "ivf_pq_serialize.cuh"
#include <cuvs/neighbors/ivf_pq.hpp>
#include <sstream>

namespace cuvs::neighbors::ivf_pq {

void deserialize_file(raft::resources const& handle,
                      const std::string& filename,
                      cuvs::neighbors::ivf_pq::index<int64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  *index = cuvs::neighbors::ivf_pq::detail::deserialize<int64_t>(handle, filename);
}

void deserialize(raft::resources const& handle,
                 const std::string& str,
                 cuvs::neighbors::ivf_pq::index<int64_t>* index)
{
  if (!index) { RAFT_FAIL("Invalid index pointer"); }
  std::istringstream is(str);
  *index = cuvs::neighbors::ivf_pq::detail::deserialize<int64_t>(handle, is);
}
}  // namespace cuvs::neighbors::ivf_pq