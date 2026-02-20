/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/registration_tags.hpp>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/MakeFragmentKey.hpp>
#include <cuvs/distance/distance.hpp>
#include <string>

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
      using CodebookTag = cuvs::neighbors::cagra::detail::tag_codebook_half;
      using QueryTag    = cuvs::neighbors::cagra::detail::tag_h;
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, CodebookTag, QueryTag>();
      key += "t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      key += "_" + params;
    } else {
      using QueryTag = cuvs::neighbors::cagra::detail::tag_f;
      key += "standard_t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, QueryTag>();
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
      std::string key   = "compute_distance_vpq_";
      using CodebookTag = cuvs::neighbors::cagra::detail::tag_codebook_half;
      using QueryTag    = cuvs::neighbors::cagra::detail::tag_h;
      auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, CodebookTag, QueryTag>();
      key += "t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      key += "_" + std::to_string(pq_bits) + "pq_" + std::to_string(pq_len) + "subd";
      key += "_" + params;
      this->device_functions.push_back(key);
    } else {
      std::string key = "compute_distance_standard_t" + std::to_string(team_size);
      key += "_dim" + std::to_string(dataset_block_dim);
      if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        using tag_uc = cuvs::neighbors::cagra::detail::tag_uc;
        if constexpr (std::is_same_v<DataTag, tag_uc>) {
          auto params = make_fragment_key<DataTag, IndexTag, DistanceTag, tag_uc>();
          key += "_" + params;
        } else {
          auto params = make_fragment_key<DataTag,
                                          IndexTag,
                                          DistanceTag,
                                          cuvs::neighbors::cagra::detail::tag_f>();
          key += "_" + params;
        }
      } else {
        auto params = make_fragment_key<DataTag,
                                        IndexTag,
                                        DistanceTag,
                                        cuvs::neighbors::cagra::detail::tag_f>();
        key += "_" + params;
      }
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
      case cuvs::distance::DistanceType::CosineExpanded:
        metric_tag = "inner_product";  // CosineExpanded uses inner_product dist_op
        break;
      case cuvs::distance::DistanceType::BitwiseHamming: metric_tag = "hamming"; break;
      default: metric_tag = "unknown"; break;
    }
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
    std::string normalization_type;
    if (metric == cuvs::distance::DistanceType::CosineExpanded) {
      normalization_type = "cosine";
    } else {
      normalization_type = "noop";
    }
    using QueryTag  = cuvs::neighbors::cagra::detail::tag_f;
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
    name += bool_to_string(topk_by_bitonic_sort) + "_";
    name += bool_to_string(bitonic_sort_and_merge_multi_warps) + "_";
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
