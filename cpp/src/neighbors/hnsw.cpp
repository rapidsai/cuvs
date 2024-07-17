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
#include <filesystem>
#include <memory>
#include <random>
#include <sys/types.h>

namespace cuvs::neighbors::hnsw {

template <typename T>
std::unique_ptr<index<T>> from_cagra(raft::resources const& res,
                                     cuvs::neighbors::cagra::index<T, uint32_t> cagra_index)
{
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0);
  auto uuid            = std::to_string(dist(rng));
  std::string filepath = "/tmp/" + uuid + ".bin";
  cuvs::neighbors::cagra::serialize_to_hnswlib(res, filepath, cagra_index);
  index<T>* hnsw_index;
  cuvs::neighbors::hnsw::deserialize(
    res, filepath, cagra_index.dim(), cagra_index.metric(), &hnsw_index);
  std::filesystem::remove(filepath);
  return std::unique_ptr<detail::index_impl<T>>(hnsw_index);
}

template <typename T, typename QueriesT>
void search(raft::resources const& res,
            const search_params& params,
            const index<T>& idx,
            raft::host_matrix_view<const QueriesT, int64_t, raft::row_major> queries,
            raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
            raft::host_matrix_view<float, int64_t, raft::row_major> distances)
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");
  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  detail::search(res, params, idx, queries, neighbors, distances);
}

template void search<float, float>(
  raft::resources const& res,
  const search_params& params,
  const index<float>& idx,
  raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
  raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
  raft::host_matrix_view<float, int64_t, raft::row_major> distances);

template void search<uint8_t, int>(
  raft::resources const& res,
  const search_params& params,
  const index<uint8_t>& idx,
  raft::host_matrix_view<const int, int64_t, raft::row_major> queries,
  raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
  raft::host_matrix_view<float, int64_t, raft::row_major> distances);

template void search<int8_t, int>(
  raft::resources const& res,
  const search_params& params,
  const index<int8_t>& idx,
  raft::host_matrix_view<const int, int64_t, raft::row_major> queries,
  raft::host_matrix_view<uint64_t, int64_t, raft::row_major> neighbors,
  raft::host_matrix_view<float, int64_t, raft::row_major> distances);

template <typename T>
void deserialize(raft::resources const& handle,
                 const std::string& filename,
                 int dim,
                 cuvs::distance::DistanceType metric,
                 index<T>** idx)
{
  *idx = new detail::index_impl<T>(filename, dim, metric);
}

}  // namespace cuvs::neighbors::hnsw
