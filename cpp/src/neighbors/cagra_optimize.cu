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

#include "cagra.cuh"
#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra {

void optimize(raft::resources const& handle,
              raft::device_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph)
{
  cuvs::neighbors::cagra::optimize<
    uint32_t,
    raft::host_device_accessor<std::experimental::default_accessor<uint32_t>,
                               raft::memory_type::device>>(handle, knn_graph, new_graph);
}
void optimize(raft::resources const& handle,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph)
{
  cuvs::neighbors::cagra::optimize<
    uint32_t,
    raft::host_device_accessor<std::experimental::default_accessor<uint32_t>,
                               raft::memory_type::host>>(handle, knn_graph, new_graph);
}

}  // namespace cuvs::neighbors::cagra