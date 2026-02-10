/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/stats/trustworthiness_score.hpp>

#include "./detail/trustworthiness_score.cuh"

namespace cuvs {  // NOLINT(modernize-concat-nested-namespaces)
namespace stats {

double trustworthiness_score(  // NOLINT(modernize-use-trailing-return-type)
  raft::resources const& handle,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X,
  raft::device_matrix_view<const float, int64_t, raft::row_major> X_embedded,
  int n_neighbors,
  cuvs::distance::DistanceType metric,
  int batch_size)
{
  RAFT_EXPECTS(X.extent(0) == X_embedded.extent(0), "Size mismatch between X and X_embedded");
  RAFT_EXPECTS(X.extent(0) <= std::numeric_limits<int>::max(), "Index type not supported");

  // TODO: Change the underlying implementation to remove the need to const_cast X_embedded.  //
  // NOLINT(google-readability-todo)
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
