/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <cuvs/distance/distance.hpp>
#include <string>

namespace cuvs {
namespace neighbors {
namespace cagra {
namespace detail {
namespace single_cta_search {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag>
struct CagraSingleCtaSearchPlanner
  : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag> {
  CagraSingleCtaSearchPlanner(cuvs::distance::DistanceType metric,
                              bool topk_by_bitonic_sort,
                              bool bitonic_sort_and_merge_multi_warps,
                              uint32_t team_size,
                              uint32_t dataset_block_dim,
                              bool is_vpq      = false,
                              uint32_t pq_bits = 0,
                              uint32_t pq_len  = 0,
                              bool persistent  = false)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag>(
        build_entrypoint_name(topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent),
        make_fragment_key<DataTag, IndexTag, DistanceTag, SourceIndexTag>())
  {
  }

 private:
  static std::string build_entrypoint_name(bool topk_by_bitonic_sort,
                                           bool bitonic_sort_and_merge_multi_warps,
                                           bool persistent)
  {
    std::string name = (persistent ? "search_single_cta_p" : "search_single_cta");
    name += (topk_by_bitonic_sort ? "_" : "_no_") + std::string("topk_by_bitonic_sort");
    name += (bitonic_sort_and_merge_multi_warps ? "_" : "_no_") +
            std::string("bitonic_sort_and_merge_multi_warps");
    return name;
  }
};

}  // namespace single_cta_search
}  // namespace detail
}  // namespace cagra
}  // namespace neighbors
}  // namespace cuvs
