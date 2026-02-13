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
      using CodebookTag = cuvs::neighbors::cagra::detail::tag_codebook_half;
      auto params       = make_fragment_key<DataTag, IndexTag, DistanceTag, CodebookTag>();
      key += metric_to_string(metric);
      key += "_t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      key += "_" + params;
    } else {
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag>();
      key += metric_to_string(metric);
      key += "_t" + std::to_string(team_size);
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
    std::string key = "compute_distance_";
    if (is_vpq) {
      key += "vpq_";
      using CodebookTag = cuvs::neighbors::cagra::detail::tag_codebook_half;
      auto params       = make_fragment_key<DataTag, IndexTag, DistanceTag, CodebookTag>();
      key += metric_to_string(metric);
      key += "_t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      key += "_" + params;
    } else {
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag>();
      key += metric_to_string(metric);
      key += "_t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + params;
    }
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
    std::string name = "search_multi_cta_kernel_";
    if (is_vpq) { name += "vpq_"; }
    name += metric_to_string(metric);
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
