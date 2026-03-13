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
        kernel_name,
        (kernel_name == "apply_filter_kernel")
          ? make_fragment_key<IndexTag, DistanceTag, SourceIndexTag>()
          : make_fragment_key<DataTag, IndexTag, DistanceTag, SourceIndexTag>())
  {
  }
};

}  // namespace multi_kernel_search
}  // namespace cuvs::neighbors::cagra::detail
