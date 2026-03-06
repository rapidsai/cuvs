/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <cuvs/distance/distance.hpp>
#include <string>

// Use nested namespace syntax to allow inclusion from within parent namespace
namespace cuvs::neighbors::cagra::detail {
namespace multi_kernel_search {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag>
struct CagraMultiKernelSearchPlanner
  : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag> {
  CagraMultiKernelSearchPlanner(cuvs::distance::DistanceType metric,
                                const std::string& kernel_name,
                                uint32_t team_size,
                                uint32_t dataset_block_dim,
                                bool is_vpq      = false,
                                uint32_t pq_bits = 0,
                                uint32_t pq_len  = 0)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag>(
        build_entrypoint_name(
          kernel_name, metric, team_size, dataset_block_dim, is_vpq, pq_bits, pq_len),
        // Special case: apply_filter_kernel doesn't use DataTag, only IndexTag, DistanceTag,
        // SourceIndexTag
        (kernel_name == "apply_filter_kernel")
          ? make_fragment_key<IndexTag, DistanceTag, SourceIndexTag>()
          : (is_vpq
               ? make_fragment_key<DataTag,
                                   IndexTag,
                                   DistanceTag,
                                   QueryTag,
                                   SourceIndexTag,
                                   CodebookTag>()
               : make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag, SourceIndexTag>()))
  {
  }

 private:
  static std::string build_entrypoint_name(const std::string& kernel_name,
                                           cuvs::distance::DistanceType metric,
                                           uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           bool is_vpq,
                                           uint32_t pq_bits,
                                           uint32_t pq_len)
  {
    if (kernel_name == "apply_filter_kernel") { return kernel_name; }

    std::string name = kernel_name;
    if (is_vpq) { name += "_vpq"; }
    name += "_team_size_" + std::to_string(team_size);
    name += "_dataset_block_dim_" + std::to_string(dataset_block_dim);
    if (is_vpq) { name += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd"; }
    return name;
  }
};

}  // namespace multi_kernel_search
}  // namespace cuvs::neighbors::cagra::detail
