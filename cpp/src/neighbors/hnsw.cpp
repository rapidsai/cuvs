/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/hnsw.hpp"

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <cstdint>
#include <sys/types.h>

namespace cuvs::neighbors::hnsw {

auto to_cagra_params(raft::matrix_extent<int64_t> dataset,
                     int M,
                     int ef_construction,
                     cuvs::distance::DistanceType metric) -> cuvs::neighbors::cagra::index_params
{
  return cuvs::neighbors::cagra::index_params::from_hnsw_soft_m(
    dataset, M, ef_construction, metric);
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
