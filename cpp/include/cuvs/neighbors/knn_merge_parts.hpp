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

#pragma once

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

namespace cuvs::neighbors {
/**
 * @brief Merge knn distances and index matrix, which have been partitioned
 * by row, into a single matrix with only the k-nearest neighbors.
 *
 * @param res raft resources
 * @param inK partitioned knn distance matrix
 * @param inV partitioned knn index matrix
 * @param outK merged knn distance matrix
 * @param outV merged knn index matrix
 * @param translations mapping of index offsets for each partition
 */
void knn_merge_parts(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t> inK,
                     raft::device_matrix_view<const int64_t, int64_t> inV,
                     raft::device_matrix_view<float, int64_t> outK,
                     raft::device_matrix_view<int64_t, int64_t> outV,
                     raft::device_vector_view<int64_t> translations);
void knn_merge_parts(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t> inK,
                     raft::device_matrix_view<const uint32_t, int64_t> inV,
                     raft::device_matrix_view<float, int64_t> outK,
                     raft::device_matrix_view<uint32_t, int64_t> outV,
                     raft::device_vector_view<uint32_t> translations);
void knn_merge_parts(raft::resources const& res,
                     raft::device_matrix_view<const float, int64_t> inK,
                     raft::device_matrix_view<const int32_t, int64_t> inV,
                     raft::device_matrix_view<float, int64_t> outK,
                     raft::device_matrix_view<int32_t, int64_t> outV,
                     raft::device_vector_view<int32_t> translations);
}  // namespace cuvs::neighbors
