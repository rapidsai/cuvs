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

#include <cuvs/neighbors/reachability.hpp>

#include "./detail/reachability.cuh"

namespace cuvs::neighbors::reachability {

void mutual_reachability_graph(const raft::resources& handle,
                               raft::device_matrix_view<const float, int, raft::row_major> X,
                               int min_samples,
                               raft::device_vector_view<int> indptr,
                               raft::device_vector_view<float> core_dists,
                               raft::sparse::COO<float, int>& out,
                               cuvs::distance::DistanceType metric,
                               float alpha)
{
  // TODO: assert core_dists/indptr have right shape
  // TODO: add test
  cuvs::neighbors::detail::reachability::mutual_reachability_graph<int, float>(
    handle,
    X.data_handle(),
    X.extent(0),
    X.extent(1),
    metric,
    min_samples,
    alpha,
    indptr.data_handle(),
    core_dists.data_handle(),
    out);
}
}  // namespace cuvs::neighbors::reachability
