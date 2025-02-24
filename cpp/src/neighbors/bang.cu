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

#include "./detail/knn_brute_force.cuh"

#include <cuvs/neighbors/bang.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <raft/core/resources.hpp>

#include <raft/core/copy.hpp>

namespace {
enum class Metric {
  L2,
  MIPS,
};
}
namespace cuvs::neighbors::experimental::bang {

template <typename T>
index<T>::index(raft::resources const& res,
                const std::string& disk_index_path,
                cuvs::distance::DistanceType metric)
  // this constructor is just for a temporary index, for use in the deserialization
  // api. all the parameters here will get replaced with loaded values - that aren't
  // necessarily known ahead of time before deserialization.
  // TODO: do we even need a handle here - could just construct one?
  : cuvs::neighbors::index(), metric_(metric)
{
  bang_instance.bang_load(disk_index_path);
}

#define CUVS_INST_BANG(T)

template <typename T>
void search(raft::resources const& res,
            const search_params& params,
            const index<T>& index,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  int numQueries = static_cast<int>(queries.extent(0));
  int k          = static_cast<int>(neighbors.extent(1));
  auto metric_enum =
    index.metric() == cuvs::distance::DistanceType::L2Expanded ? Metric::L2 : Metric::MIPS;
  index.bang_search_instance.bang_set_searchparams(k, params.worklist_length, metric_enum);
  index.bang_search_instance.bang_alloc(res, numQueries);
  index.bang_search_instance.bang_init(res, numQueries);
  index.bang_search_instance.bang_query(
    res, queries, queries.extent(0), neighbors.data_handle(), distances.data_handle());
  // bang_search_instance.bang_free();
}

CUVS_INST_BANG(float);
CUVS_INST_BANG(int8_t);
CUVS_INST_BANG(uint8_t);

#undef CUVS_INST_BANG

}  // namespace cuvs::neighbors::experimental::bang
