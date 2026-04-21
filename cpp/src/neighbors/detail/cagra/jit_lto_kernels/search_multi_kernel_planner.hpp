/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/distance/distance.hpp>
#include <string>

namespace cuvs::neighbors::cagra::detail::multi_kernel_search {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag>
struct CagraMultiKernelSearchPlanner
  : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag> {
  static inline LauncherJitCache launcher_jit_cache{};

  CagraMultiKernelSearchPlanner(cuvs::distance::DistanceType /*metric*/,
                                const std::string& kernel_name,
                                uint32_t /*team_size*/,
                                uint32_t /*dataset_block_dim*/,
                                bool /*is_vpq*/,
                                uint32_t /*pq_bits*/,
                                uint32_t /*pq_len*/)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag>(kernel_name,
                                                                              launcher_jit_cache)
  {
  }

  void add_linked_kernel(std::string const& kernel_name)
  {
    if (kernel_name == "random_pickup") {
      this->template add_static_fragment<
        fragment_tag_random_pickup<DataTag, IndexTag, DistanceTag>>();
    } else if (kernel_name == "compute_distance_to_child_nodes") {
      this->template add_static_fragment<
        fragment_tag_compute_distance_to_child_nodes<DataTag,
                                                     IndexTag,
                                                     DistanceTag,
                                                     SourceIndexTag>>();
    } else if (kernel_name == "apply_filter_kernel") {
      this->template add_static_fragment<
        fragment_tag_apply_filter_kernel<IndexTag, DistanceTag, SourceIndexTag>>();
    } else {
      RAFT_FAIL("Unknown CAGRA multi-kernel JIT kernel: %s", kernel_name.c_str());
    }
  }
};

}  // namespace cuvs::neighbors::cagra::detail::multi_kernel_search
