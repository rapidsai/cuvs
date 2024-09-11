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

#include <cuvs/stats/trustworthiness_score.hpp>

#include "./detail/trustworthiness_score.cuh"

namespace cuvs {
namespace stats {

double trustworthiness_score(
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X_embedded,
  int n_neighbors,
  cuvs::distance::DistanceType metric,
  int batch_size)
{
  RAFT_EXPECTS(X.extent(0) == X_embedded.extent(0), "Size mismatch between X and X_embedded");
  RAFT_EXPECTS(X.extent(0) <= std::numeric_limits<int>::max(), "Index type not supported");

  // TODO: Change the underlying implementation to remove the need to const_cast X_embedded.
  return detail::trustworthiness_score<float>(handle,
                                              X.data_handle(),
                                              const_cast<float*>(X_embedded.data_handle()),
                                              metric,
                                              static_cast<int>(X.extent(0)),
                                              static_cast<int>(X.extent(1)),
                                              static_cast<int>(X_embedded.extent(1)),
                                              n_neighbors,
                                              batch_size);
}

}  // namespace stats
}  // namespace cuvs
