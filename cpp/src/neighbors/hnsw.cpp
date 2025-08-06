/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

auto to_cagra_params(raft::matrix_extent<int64_t> dataset,
                     int M,
                     int ef_construction,
                     cuvs::distance::DistanceType metric) -> cuvs::neighbors::cagra::index_params
{
  auto ivf_pq_params = cuvs::neighbors::graph_build_params::ivf_pq_params(dataset, metric);
  ivf_pq_params.search_params.n_probes =
    std::round(std::sqrt(ivf_pq_params.build_params.n_lists) / 20 + ef_construction / 16);

  cagra::index_params params;
  params.graph_build_params        = ivf_pq_params;
  params.graph_degree              = M * 2;
  params.intermediate_graph_degree = M * 3;

  return params;
}

#define CUVS_INST_HNSW_FROM_CAGRA(T)                                                  \
  std::unique_ptr<index<T>> from_cagra(                                               \
    raft::resources const& res,                                                       \
    const index_params& params,                                                       \
    const cuvs::neighbors::cagra::index<T, uint32_t>& cagra_index,                    \
    std::optional<raft::host_matrix_view<const T, int64_t, raft::row_major>> dataset) \
  {                                                                                   \
    return detail::from_cagra<T>(res, params, cagra_index, dataset);                  \
  }

CUVS_INST_HNSW_FROM_CAGRA(float);
CUVS_INST_HNSW_FROM_CAGRA(half);
CUVS_INST_HNSW_FROM_CAGRA(uint8_t);
CUVS_INST_HNSW_FROM_CAGRA(int8_t);

#undef CUVS_INST_HNSW_FROM_CAGRA

#define CUVS_INST_HNSW_EXTEND(T)                                                            \
  void extend(raft::resources const& res,                                                   \
              const extend_params& params,                                                  \
              raft::host_matrix_view<const T, int64_t, raft::row_major> additional_dataset, \
              index<T>& idx)                                                                \
  {                                                                                         \
    detail::extend<T>(res, params, additional_dataset, idx);                                \
  }

CUVS_INST_HNSW_EXTEND(float);
CUVS_INST_HNSW_EXTEND(half);
CUVS_INST_HNSW_EXTEND(uint8_t);
CUVS_INST_HNSW_EXTEND(int8_t);

#undef CUVS_INST_HNSW_EXTEND

#define CUVS_INST_HNSW_SEARCH(T)                                                    \
  void search(raft::resources const& res,                                           \
              const search_params& params,                                          \
              const index<T>& idx,                                                  \
              raft::host_matrix_view<const T, int64_t, raft::row_major> queries,    \
              raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors, \
              raft::host_matrix_view<float, int64_t, raft::row_major> distances)    \
  {                                                                                 \
    detail::search<T>(res, params, idx, queries, neighbors, distances);             \
  }

CUVS_INST_HNSW_SEARCH(float);
CUVS_INST_HNSW_SEARCH(half);
CUVS_INST_HNSW_SEARCH(uint8_t);
CUVS_INST_HNSW_SEARCH(int8_t);

#undef CUVS_INST_HNSW_SEARCH

#define CUVS_INST_HNSW_SERIALIZE(T)                                                            \
  void serialize(raft::resources const& res, const std::string& filename, const index<T>& idx) \
  {                                                                                            \
    detail::serialize<T>(res, filename, idx);                                                  \
  }                                                                                            \
  void deserialize(raft::resources const& res,                                                 \
                   const index_params& params,                                                 \
                   const std::string& filename,                                                \
                   int dim,                                                                    \
                   cuvs::distance::DistanceType metric,                                        \
                   index<T>** idx)                                                             \
  {                                                                                            \
    detail::deserialize<T>(res, params, filename, dim, metric, idx);                           \
  }

CUVS_INST_HNSW_SERIALIZE(float);
CUVS_INST_HNSW_SERIALIZE(half);
CUVS_INST_HNSW_SERIALIZE(uint8_t);
CUVS_INST_HNSW_SERIALIZE(int8_t);

#undef CUVS_INST_HNSW_SERIALIZE

}  // namespace cuvs::neighbors::hnsw
