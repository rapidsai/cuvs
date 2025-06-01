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

namespace helpers {
void build_single_linkage_dendrogram(raft::resources const& handle,
                                     raft::device_matrix_view<const float, int, raft::row_major> X,
                                     cuvs::distance::DistanceType metric,
                                     raft::device_vector_view<int, int> graph_indptr,
                                     raft::device_coo_matrix_view<float, int, int, size_t> graph,
                                     raft::device_vector_view<float, int> core_dists,
                                     raft::device_coo_matrix_view<float, int, int, int> out_mst,
                                     raft::device_matrix_view<int, int> dendrogram,
                                     raft::device_vector_view<float, int> out_distances,
                                     raft::device_vector_view<int, int> out_sizes)
{
  cuvs::neighbors::detail::reachability::build_single_linkage_dendrogram(
    handle,
    X.data_handle(),
    static_cast<size_t>(X.extent(0)),
    static_cast<size_t>(X.extent(1)),
    metric,
    graph_indptr.data_handle(),
    graph,
    core_dists.data_handle(),
    out_mst.structure_view().get_rows().data(),
    out_mst.structure_view().get_cols().data(),
    out_mst.get_elements().data(),
    dendrogram.data_handle(),
    out_distances.data_handle(),
    out_sizes.data_handle());
}
}  // namespace helpers
}  // namespace cuvs::neighbors::reachability
