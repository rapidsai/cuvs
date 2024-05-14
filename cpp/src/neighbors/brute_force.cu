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

#include "./detail/knn_brute_force.cuh"
#include <cuvs/neighbors/brute_force.hpp>

namespace cuvs::neighbors::brute_force {
#define CUVS_INST_BFKNN(T)                                                           \
  auto build(raft::resources const& res,                                             \
             raft::device_matrix_view<const T, int64_t, raft::row_major> dataset,    \
             cuvs::distance::DistanceType metric,                                    \
             T metric_arg)                                                           \
    ->cuvs::neighbors::brute_force::index<T>                                         \
  {                                                                                  \
    return detail::build<T>(res, dataset, metric, metric_arg);                       \
  }                                                                                  \
                                                                                     \
  void search(raft::resources const& res,                                            \
              const cuvs::neighbors::brute_force::index<T>& idx,                     \
              raft::device_matrix_view<const T, int64_t, raft::row_major> queries,   \
              raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors, \
              raft::device_matrix_view<T, int64_t, raft::row_major> distances)       \
  {                                                                                  \
    detail::brute_force_search<T, int64_t>(res, idx, queries, neighbors, distances); \
  }                                                                                  \
                                                                                     \
  template struct cuvs::neighbors::brute_force::index<T>;

CUVS_INST_BFKNN(float);

#undef CUVS_INST_BFKNN

}  // namespace cuvs::neighbors::brute_force
