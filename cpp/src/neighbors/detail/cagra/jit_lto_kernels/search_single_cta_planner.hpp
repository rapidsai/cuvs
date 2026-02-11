/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/cagra/search_single_cta_tags.hpp>
#include <cuvs/distance/distance.hpp>
#include <iostream>
#include <string>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

template <typename DataTag, typename IndexTag, typename DistanceTag, typename SourceIndexTag>
struct CagraSearchPlanner : AlgorithmPlanner {
  CagraSearchPlanner(bool topk_by_bitonic_sort, bool bitonic_sort_and_merge_multi_warps)
    : AlgorithmPlanner("search_single_cta_kernel_" + bool_to_string(topk_by_bitonic_sort) + "_" +
                         bool_to_string(bitonic_sort_and_merge_multi_warps),
                       make_fragment_key<DataTag, IndexTag, DistanceTag, SourceIndexTag>())
  {
  }

  void add_setup_workspace_device_function(cuvs::distance::DistanceType metric,
                                           uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           bool is_vpq,
                                           uint32_t pq_bits = 0,
                                           uint32_t pq_len  = 0)
  {
    std::string key = "setup_workspace_";
    key += metric_to_string(metric);
    key += "_t" + std::to_string(team_size);
    key += "_dim" + std::to_string(dataset_block_dim);
    if (is_vpq) { key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd"; }
    this->device_functions.push_back(key);
  }

  void add_compute_distance_device_function(cuvs::distance::DistanceType metric,
                                            uint32_t team_size,
                                            uint32_t dataset_block_dim,
                                            bool is_vpq,
                                            uint32_t pq_bits = 0,
                                            uint32_t pq_len  = 0)
  {
    std::string key = "compute_distance_";
    key += metric_to_string(metric);
    key += "_t" + std::to_string(team_size);
    key += "_dim" + std::to_string(dataset_block_dim);
    if (is_vpq) { key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd"; }
    this->device_functions.push_back(key);
  }

  void add_sample_filter_device_function(std::string filter_name)
  {
    this->device_functions.push_back("sample_filter_" + filter_name);
  }

 private:
  static std::string bool_to_string(bool b) { return b ? "true" : "false"; }

  static std::string metric_to_string(cuvs::distance::DistanceType metric)
  {
    switch (metric) {
      case cuvs::distance::DistanceType::L2Expanded:
      case cuvs::distance::DistanceType::L2Unexpanded: return "L2Expanded";
      case cuvs::distance::DistanceType::InnerProduct: return "InnerProduct";
      case cuvs::distance::DistanceType::CosineExpanded: return "CosineExpanded";
      case cuvs::distance::DistanceType::BitwiseHamming: return "BitwiseHamming";
      default: return "Unknown";
    }
  }
};

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
