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

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/sparse/coo.hpp>

#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::reachability {

/**
 * @defgroup reachability_cpp Mutual Reachability
 * @{
 */
/**
 * Constructs a mutual reachability graph, which is a k-nearest neighbors
 * graph projected into mutual reachability space using the following
 * function for each data point, where core_distance is the distance
 * to the kth neighbor: max(core_distance(a), core_distance(b), d(a, b))
 *
 * Unfortunately, points in the tails of the pdf (e.g. in sparse regions
 * of the space) can have very large neighborhoods, which will impact
 * nearby neighborhoods. Because of this, it's possible that the
 * radius for points in the main mass, which might have a very small
 * radius initially, to expand very large. As a result, the initial
 * knn which was used to compute the core distances may no longer
 * capture the actual neighborhoods after projection into mutual
 * reachability space.
 *
 * For the experimental version, we execute the knn twice- once
 * to compute the radii (core distances) and again to capture
 * the final neighborhoods. Future iterations of this algorithm
 * will work improve upon this "exact" version, by using
 * more specialized data structures, such as space-partitioning
 * structures. It has also been shown that approximate nearest
 * neighbors can yield reasonable neighborhoods as the
 * data sizes increase.
 *
 * @param[in] handle raft handle for resource reuse
 * @param[in] X input data points (size m * n)
 * @param[in] min_samples this neighborhood will be selected for core distances
 * @param[out] indptr CSR indptr of output knn graph (size m + 1)
 * @param[out] core_dists output core distances array (size m)
 * @param[out] out COO object, uninitialized on entry, on exit it stores the
 *             (symmetrized) maximum reachability distance for the k nearest
 *             neighbors.
 * @param[in] metric distance metric to use, default Euclidean
 * @param[in] alpha weight applied when internal distance is chosen for
 *            mutual reachability (value of 1.0 disables the weighting)
 */
void mutual_reachability_graph(
  const raft::resources& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  int min_samples,
  raft::device_vector_view<int64_t> indptr,
  raft::device_vector_view<float> core_dists,
  raft::sparse::COO<float, int64_t>& out,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtExpanded,
  float alpha                         = 1.0);
/**
 * @}
 */
}  // namespace cuvs::neighbors::reachability
