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
        build_entrypoint_name(metric,
                              topk_by_bitonic_sort,
                              bitonic_sort_and_merge_multi_warps,
                              team_size,
                              dataset_block_dim,
                              is_vpq,
                              pq_bits,
                              pq_len,
                              persistent),
        is_vpq ? make_fragment_key<DataTag, IndexTag, DistanceTag, SourceIndexTag, CodebookTag>()
               : make_fragment_key<DataTag, IndexTag, DistanceTag, SourceIndexTag>())
  {
  }

 private:
  static std::string build_entrypoint_name(cuvs::distance::DistanceType metric,
                                           bool topk_by_bitonic_sort,
                                           bool bitonic_sort_and_merge_multi_warps,
                                           uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           bool is_vpq,
                                           uint32_t pq_bits,
                                           uint32_t pq_len,
                                           bool persistent)
  {
    std::string name = (persistent ? "search_single_cta_kernel_p" : "search_single_cta_kernel");
    name += std::string(topk_by_bitonic_sort ? "_" : "_no_") + "topk_by_bitonic_sort";
    name += std::string(bitonic_sort_and_merge_multi_warps ? "_" : "_no_") +
            "bitonic_sort_and_merge_multi_warps";
    name += "_team_size_" + std::to_string(team_size);
    name += "_dataset_block_dim_" + std::to_string(dataset_block_dim);
    if (is_vpq) { name += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd"; }
    return name;
  }
};

}  // namespace single_cta_search
}  // namespace detail
}  // namespace cagra
}  // namespace neighbors
}  // namespace cuvs
