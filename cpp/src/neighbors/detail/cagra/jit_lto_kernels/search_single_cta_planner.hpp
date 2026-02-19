/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Include tags header before namespace (it defines a namespace)
#include <cuvs/detail/jit_lto/cagra/search_single_cta_tags.hpp>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/distance/distance.hpp>
#include <iostream>
#include <raft/core/logger.hpp>
#include <string>

// Use nested namespace syntax to allow inclusion from within parent namespace
namespace cuvs {
namespace neighbors {
namespace cagra {
namespace detail {
namespace single_cta_search {

template <typename DataTag, typename IndexTag, typename DistanceTag, typename SourceIndexTag>
struct CagraSearchPlanner : AlgorithmPlanner {
  CagraSearchPlanner(cuvs::distance::DistanceType metric,
                     bool topk_by_bitonic_sort,
                     bool bitonic_sort_and_merge_multi_warps,
                     uint32_t team_size,
                     uint32_t dataset_block_dim,
                     bool is_vpq      = false,
                     uint32_t pq_bits = 0,
                     uint32_t pq_len  = 0,
                     bool persistent  = false)
    : AlgorithmPlanner(build_entrypoint_name(metric,
                                             topk_by_bitonic_sort,
                                             bitonic_sort_and_merge_multi_warps,
                                             team_size,
                                             dataset_block_dim,
                                             is_vpq,
                                             pq_bits,
                                             pq_len,
                                             persistent),
                       is_vpq
                         ? make_fragment_key<DataTag,
                                             IndexTag,
                                             DistanceTag,
                                             SourceIndexTag,
                                             cuvs::neighbors::cagra::detail::tag_codebook_half>()
                         : make_fragment_key<DataTag, IndexTag, DistanceTag, SourceIndexTag>())
  {
    std::string kernel_type = persistent ? "persistent" : "regular";
    std::cerr << "[JIT] CagraSearchPlanner created for " << kernel_type
              << " JIT kernel (topk_by_bitonic_sort=" << bool_to_string(topk_by_bitonic_sort)
              << ", bitonic_sort_and_merge_multi_warps="
              << bool_to_string(bitonic_sort_and_merge_multi_warps)
              << ", metric=" << metric_to_string(metric) << ")" << std::endl;
    std::cerr.flush();
  }

  void add_setup_workspace_device_function(cuvs::distance::DistanceType metric,
                                           uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           bool is_vpq,
                                           uint32_t pq_bits = 0,
                                           uint32_t pq_len  = 0)
  {
    std::string key = "setup_workspace_";
    if (is_vpq) {
      key += "vpq_";
      // For VPQ, include codebook type tag in template parameters
      // Note: Metric is no longer in the key - VPQ only supports L2Expanded
      using CodebookTag = cuvs::neighbors::cagra::detail::tag_codebook_half;
      // Use template tags only for types, strings for integers/enums
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, CodebookTag>();
      key += "t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      key += "_" + params;
    } else {
      // Standard dataset - Metric is no longer in the key, linked via dist_op and normalization
      // fragments Use template tags only for types, strings for integers/enums
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag>();
      key += "standard_t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + params;
    }
    this->device_functions.push_back(key);
  }

  void add_compute_distance_device_function(cuvs::distance::DistanceType metric,
                                            uint32_t team_size,
                                            uint32_t dataset_block_dim,
                                            bool is_vpq,
                                            uint32_t pq_bits = 0,
                                            uint32_t pq_len  = 0)
  {
    if (is_vpq) {
      // VPQ: Metric is no longer in the key - VPQ only supports L2Expanded
      std::string key   = "compute_distance_vpq_";
      using CodebookTag = cuvs::neighbors::cagra::detail::tag_codebook_half;
      auto params       = make_fragment_key<DataTag, IndexTag, DistanceTag, CodebookTag>();
      key += "t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      key += "_" + params;
      this->device_functions.push_back(key);
    } else {
      // Standard: compute_distance_standard no longer has metric in the name
      // Metric is handled via dist_op fragments
      std::string key = "compute_distance_standard_t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag>();
      key += "_" + params;
      this->device_functions.push_back(key);

      // Add dist_op fragment for the metric
      add_dist_op_device_function(metric);

      // Add normalization fragment (cosine or noop)
      add_normalization_device_function(metric, team_size, dataset_block_dim);
    }
  }

  void add_dist_op_device_function(cuvs::distance::DistanceType metric)
  {
    std::string metric_tag;
    switch (metric) {
      case cuvs::distance::DistanceType::L2Expanded:
      case cuvs::distance::DistanceType::L2Unexpanded: metric_tag = "l2"; break;
      case cuvs::distance::DistanceType::InnerProduct: metric_tag = "inner_product"; break;
      case cuvs::distance::DistanceType::CosineExpanded:
        metric_tag = "inner_product";  // CosineExpanded uses inner_product dist_op
        break;
      case cuvs::distance::DistanceType::BitwiseHamming: metric_tag = "hamming"; break;
      default: metric_tag = "unknown"; break;
    }
    auto params     = make_fragment_key<DataTag, DistanceTag>();
    std::string key = "dist_op_" + metric_tag + "_" + params;
    this->device_functions.push_back(key);
  }

  void add_normalization_device_function(cuvs::distance::DistanceType metric,
                                         uint32_t team_size,
                                         uint32_t dataset_block_dim)
  {
    // Both cosine and noop fragments provide the same function name "apply_normalization_standard"
    // but register with different fragment names. The planner links the appropriate one based on
    // metric.
    std::string normalization_type;
    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      normalization_type = "cosine";
    } else {
      normalization_type = "noop";
    }
    auto params     = make_fragment_key<DataTag, IndexTag, DistanceTag>();
    std::string key = "apply_normalization_standard_" + normalization_type;
    key += "_t" + std::to_string(team_size);
    key += "_dim" + std::to_string(dataset_block_dim);
    key += "_" + params;
    this->device_functions.push_back(key);
  }

  void add_sample_filter_device_function(std::string filter_name)
  {
    this->device_functions.push_back("sample_filter_" + filter_name);
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
    std::string name = (persistent ? "search_single_cta_kernel_p_" : "search_single_cta_kernel_");
    if (is_vpq) {
      name += "vpq_";
      // Note: Metric is no longer in VPQ kernel names - VPQ only supports L2Expanded
    }
    name += bool_to_string(topk_by_bitonic_sort) + "_";
    name += bool_to_string(bitonic_sort_and_merge_multi_warps) + "_";
    // Note: Metric is no longer in kernel names - it's linked via dist_op and normalization
    // fragments
    name += "t" + std::to_string(team_size);
    name += "_dim" + std::to_string(dataset_block_dim);
    if (is_vpq) { name += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd"; }
    return name;
  }

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

}  // namespace single_cta_search
}  // namespace detail
}  // namespace cagra
}  // namespace neighbors
}  // namespace cuvs
