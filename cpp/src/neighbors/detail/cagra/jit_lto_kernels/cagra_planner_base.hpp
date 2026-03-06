/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/logger.hpp>
#include <string>

namespace cuvs::neighbors::cagra::detail {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          typename CodebookTag>
struct CagraPlannerBase : AlgorithmPlanner {
  using AlgorithmPlanner::device_functions;

  CagraPlannerBase(const std::string& entrypoint, const std::string& params)
    : AlgorithmPlanner(entrypoint, params)
  {
  }
  void add_setup_workspace_device_function(cuvs::distance::DistanceType metric,
                                           uint32_t team_size,
                                           uint32_t dataset_block_dim,
                                           bool is_vpq,
                                           uint32_t pq_bits = 0,
                                           uint32_t pq_len  = 0)
  {
    std::string key = "setup_workspace";
    if (is_vpq) {
      key += "_vpq";
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag>();
      key += "_team_size_" + std::to_string(team_size);
      key += "_dataset_block_dim_" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      if (!params.empty()) { key += "_" + params; }
    } else {
      key += "_standard_team_size_" + std::to_string(team_size);
      key += "_dataset_block_dim_" + std::to_string(dataset_block_dim);
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag>();
      if (!params.empty()) { key += "_" + params; }
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
      std::string key = "compute_distance_vpq";
      auto params     = make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag>();
      key += "_team_size_" + std::to_string(team_size);
      key += "_dataset_block_dim_" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      if (!params.empty()) { key += "_" + params; }
      this->device_functions.push_back(key);
    } else {
      std::string key = "compute_distance_standard_team_size_" + std::to_string(team_size);
      key += "_dataset_block_dim_" + std::to_string(dataset_block_dim);
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag>();
      if (!params.empty()) { key += "_" + params; }
      this->device_functions.push_back(key);
      add_dist_op_device_function(metric);
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
      case cuvs::distance::DistanceType::CosineExpanded: metric_tag = "inner_product"; break;
      case cuvs::distance::DistanceType::BitwiseHamming: metric_tag = "hamming"; break;
      default: metric_tag = "unknown"; break;
    }
    auto params     = make_fragment_key<QueryTag, DistanceTag>();
    std::string key = "dist_op_" + metric_tag;
    if (!params.empty()) { key += "_" + params; }
    this->device_functions.push_back(key);
  }

  void add_normalization_device_function(cuvs::distance::DistanceType metric,
                                         uint32_t team_size,
                                         uint32_t dataset_block_dim)
  {
    std::string normalization_type;
    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      normalization_type = "cosine";
    } else {
      normalization_type = "noop";
    }
    auto params     = make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag>();
    std::string key = "apply_normalization_standard_" + normalization_type;
    key += "_team_size_" + std::to_string(team_size);
    key += "_dataset_block_dim_" + std::to_string(dataset_block_dim);
    if (!params.empty()) { key += "_" + params; }
    this->device_functions.push_back(key);
  }

  void add_sample_filter_device_function(std::string filter_name)
  {
    this->device_functions.push_back("sample_filter_" + filter_name);
  }
};

}  // namespace cuvs::neighbors::cagra::detail
