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
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace cuvs {
namespace stats {
/**
 * @defgroup stats_trustworthiness Trustworthiness
 * @{
 */

/**
 * @brief Compute the trustworthiness score
 * @param[in] handle the raft handle
 * @param[in] X: Data in original dimension
 * @param[in] X_embedded: Data in target dimension (embedding)
 * @param[in] n_neighbors Number of neighbors considered by trustworthiness score
 * @param[in] metric Distance metric to use. Euclidean (L2) is used by default
 * @param[in] batch_size Batch size
 * @return Trustworthiness score
 * @note The constness of the data in X_embedded is currently casted away and the data is slightly
 * modified.
 */
double trustworthiness_score(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X_embedded,
  int n_neighbors,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2SqrtUnexpanded,
  int batch_size                      = 512);

/** @} */  // end group stats_trustworthiness
}  // namespace stats
}  // namespace cuvs
