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

#ifdef CUVS_BUILD_MG_ALGOS

#include <cuvs/neighbors/common.hpp>
#include <raft/core/device_resources_snmg.hpp>

#define DEFAULT_SEARCH_BATCH_SIZE 1 << 20

/// \defgroup mg_cpp_index_params ANN MG index build parameters

namespace cuvs::neighbors::mg {
/** Distribution mode */
/// \ingroup mg_cpp_index_params
enum distribution_mode {
  /** Index is replicated on each device, favors throughput */
  REPLICATED,
  /** Index is split on several devices, favors scaling */
  SHARDED
};

/// \defgroup mg_cpp_search_params ANN MG search parameters

/** Search mode when using a replicated index */
/// \ingroup mg_cpp_search_params
enum replicated_search_mode {
  /** Search queries are splited to maintain equal load on GPUs */
  LOAD_BALANCER,
  /** Each search query is processed by a single GPU in a round-robin fashion */
  ROUND_ROBIN
};

/** Merge mode when using a sharded index */
/// \ingroup mg_cpp_search_params
enum sharded_merge_mode {
  /** Search batches are merged on the root rank */
  MERGE_ON_ROOT_RANK,
  /** Search batches are merged in a tree reduction fashion */
  TREE_MERGE
};

/** Build parameters */
/// \ingroup mg_cpp_index_params
template <typename Upstream>
struct index_params : public Upstream {
  index_params() : mode(SHARDED) {}

  index_params(const Upstream& sp) : Upstream(sp), mode(SHARDED) {}

  /** Distribution mode */
  cuvs::neighbors::mg::distribution_mode mode = SHARDED;
};

/** Search parameters */
/// \ingroup mg_cpp_search_params
template <typename Upstream>
struct search_params : public Upstream {
  search_params() : search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE) {}

  search_params(const Upstream& sp)
    : Upstream(sp), search_mode(LOAD_BALANCER), merge_mode(TREE_MERGE)
  {
  }

  /** Replicated search mode */
  cuvs::neighbors::mg::replicated_search_mode search_mode = LOAD_BALANCER;
  /** Sharded merge mode */
  cuvs::neighbors::mg::sharded_merge_mode merge_mode = TREE_MERGE;
};

template <typename AnnIndexType, typename T, typename IdxT>
struct index {
  index(distribution_mode mode, int num_ranks_);
  index(const raft::device_resources_snmg& clique, const std::string& filename);

  index(const index&)                    = delete;
  index(index&&)                         = default;
  auto operator=(const index&) -> index& = delete;
  auto operator=(index&&) -> index&      = default;

  distribution_mode mode_;
  int num_ranks_;
  std::vector<iface<AnnIndexType, T, IdxT>> ann_interfaces_;

  // for load balancing mechanism
  std::shared_ptr<std::atomic<int64_t>> round_robin_counter_;
};

}  // namespace cuvs::neighbors::mg

#else

static_assert(false,
              "FORBIDEN_MG_ALGORITHM_IMPORT\n\n"
              "Please recompile the cuVS library with MG algorithms BUILD_MG_ALGOS=ON.\n");

#endif
