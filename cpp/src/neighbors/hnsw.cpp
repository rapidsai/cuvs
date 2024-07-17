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

#include "detail/hnsw.hpp"
#include <cstdint>
#include <cuvs/neighbors/hnsw.hpp>
#include <sys/types.h>

namespace cuvs::neighbors::hnsw {

#define CUVS_INST_HNSW_FROM_CAGRA(T)                                                           \
  std::unique_ptr<index<T>> from_cagra(                                                        \
    raft::resources const& res, const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index) \
  {                                                                                            \
    return detail::from_cagra<T>(res, cagra_index);                                            \
  }

CUVS_INST_HNSW_FROM_CAGRA(float);
CUVS_INST_HNSW_FROM_CAGRA(uint8_t);
CUVS_INST_HNSW_FROM_CAGRA(int8_t);

#undef CUVS_INST_HNSW_FROM_CAGRA

#define CUVS_INST_HNSW_SEARCH(T, QueriesT)                                              \
  void search(raft::resources const& res,                                               \
              const search_params& params,                                              \
              const index<T>& idx,                                                      \
              raft::host_matrix_view<const QueriesT, int64_t, raft::row_major> queries, \
              raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,     \
              raft::host_matrix_view<float, int64_t, raft::row_major> distances)        \
  {                                                                                     \
    detail::search<T, QueriesT>(res, params, idx, queries, neighbors, distances);       \
  }

CUVS_INST_HNSW_SEARCH(float, float);
CUVS_INST_HNSW_SEARCH(uint8_t, int);
CUVS_INST_HNSW_SEARCH(int8_t, int);

#undef CUVS_INST_HNSW_SEARCH

#define CUVS_INST_HNSW_DESERIALIZE(T)                        \
  void deserialize(raft::resources const& res,               \
                   const std::string& filename,              \
                   int dim,                                  \
                   cuvs::distance::DistanceType metric,      \
                   index<T>** idx)                           \
  {                                                          \
    detail::deserialize<T>(res, filename, dim, metric, idx); \
  }

CUVS_INST_HNSW_DESERIALIZE(float);
CUVS_INST_HNSW_DESERIALIZE(uint8_t);
CUVS_INST_HNSW_DESERIALIZE(int8_t);

#undef CUVS_INST_HNSW_DESERIALIZE

}  // namespace cuvs::neighbors::hnsw
