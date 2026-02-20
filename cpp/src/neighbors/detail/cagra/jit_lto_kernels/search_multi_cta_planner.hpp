/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Include tags header before namespace (it defines a namespace)
#include <cuvs/detail/jit_lto/cagra/search_single_cta_tags.hpp>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/logger.hpp>
#include <string>

// Use nested namespace syntax to allow inclusion from within parent namespace
namespace cuvs {
namespace neighbors {
namespace cagra {
namespace detail {
namespace multi_cta_search {

template <typename DataTag, typename IndexTag, typename DistanceTag, typename SourceIndexTag>
struct CagraMultiCtaSearchPlanner : AlgorithmPlanner {
  CagraMultiCtaSearchPlanner(cuvs::distance::DistanceType metric,
                             uint32_t team_size,
                             uint32_t dataset_block_dim,
                             bool is_vpq      = false,
                             uint32_t pq_bits = 0,
                             uint32_t pq_len  = 0)
    : AlgorithmPlanner(
        build_entrypoint_name(metric, team_size, dataset_block_dim, is_vpq, pq_bits, pq_len),
        is_vpq ? make_fragment_key<DataTag,
                                   IndexTag,
                                   DistanceTag,
                                   SourceIndexTag,
                                   cuvs::neighbors::cagra::detail::tag_codebook_half>()
               : make_fragment_key<DataTag, IndexTag, DistanceTag, SourceIndexTag>()),
      entrypoint_name_(
        build_entrypoint_name(metric, team_size, dataset_block_dim, is_vpq, pq_bits, pq_len))
  {
  }

  const std::string& get_entrypoint_name() const { return entrypoint_name_; }

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
      // Note: Metric is no longer in the key - VPQ only supports L2Expanded
      using CodebookTag = cuvs::neighbors::cagra::detail::tag_codebook_half;
      auto params       = make_fragment_key<DataTag, IndexTag, DistanceTag, CodebookTag>();
      key += "t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      key += "_" + params;
    } else {
      // Standard dataset - Metric is no longer in the key, linked via dist_op and normalization
      // fragments
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
    // QueryT can be float (for most metrics) or uint8_t (for BitwiseHamming)
    // DistanceT is always float
    std::string params;
    if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
      params = make_fragment_key<cuvs::neighbors::cagra::detail::tag_uc, DistanceTag>();
    } else {
      params = make_fragment_key<cuvs::neighbors::cagra::detail::tag_f, DistanceTag>();
    }
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
    // QueryT is always float for normalization (only used for CosineExpanded which uses float
    // queries)
    using QueryTag  = cuvs::neighbors::cagra::detail::tag_f;  // Always float for normalization
    auto params     = make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag>();
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
  std::string entrypoint_name_;

  static std::string build_entrypoint_name(cuvs::distance::DistanceType metric,
                                           uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           bool is_vpq,
                                           uint32_t pq_bits,
                                           uint32_t pq_len)
  {
    std::string name = "search_multi_cta_kernel";
    if (is_vpq) {
      name += "_vpq";
      // Note: Metric is no longer in VPQ kernel names - VPQ only supports L2Expanded
    }
    // Note: Metric is no longer in kernel names - it's linked via dist_op and normalization
    // fragments
    name += "_t" + std::to_string(team_size);
    name += "_dim" + std::to_string(dataset_block_dim);
    if (is_vpq) { name += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd"; }
    return name;
  }

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

}  // namespace multi_cta_search
}  // namespace detail
}  // namespace cagra
}  // namespace neighbors
}  // namespace cuvs
