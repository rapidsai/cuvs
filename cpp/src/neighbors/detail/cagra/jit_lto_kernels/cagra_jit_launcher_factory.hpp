/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUVS_ENABLE_JIT_LTO
#error "cagra_jit_launcher_factory.hpp included but CUVS_ENABLE_JIT_LTO not defined!"
#endif

#include "../compute_distance.hpp"
#include "../shared_launcher_jit.hpp"
#include "search_multi_cta_planner.hpp"
#include "search_multi_kernel_planner.hpp"
#include "search_single_cta_planner.hpp"

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/distance/distance.hpp>

#include <memory>
#include <string>

namespace cuvs::neighbors::cagra::detail {

/// Build a JIT AlgorithmLauncher for single-CTA CAGRA search (runtime VPQ / metric → tag dispatch).
template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> make_cagra_single_cta_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  bool topk_by_bitonic_sort,
  bool bitonic_sort_and_merge_multi_warps,
  bool persistent,
  const std::string& filter_name)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    single_cta_search::
      CagraSingleCtaSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                topk_by_bitonic_sort,
                bitonic_sort_and_merge_multi_warps,
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len,
                persistent);

    planner.template add_setup_workspace_device_function<DataTag,
                                                         IndexTag,
                                                         DistTag,
                                                         QueryTag,
                                                         CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.template add_compute_distance_device_function<DataTag,
                                                          IndexTag,
                                                          DistTag,
                                                          QueryTag,
                                                          CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.add_search_kernel_fragment(
      topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
    planner.add_sample_filter_device_function(filter_name);
    return planner.get_launcher();
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    single_cta_search::
      CagraSingleCtaSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                topk_by_bitonic_sort,
                bitonic_sort_and_merge_multi_warps,
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len,
                persistent);

    planner.template add_setup_workspace_device_function<DataTag,
                                                         IndexTag,
                                                         DistTag,
                                                         QueryTag,
                                                         CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.template add_compute_distance_device_function<DataTag,
                                                          IndexTag,
                                                          DistTag,
                                                          QueryTag,
                                                          CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.add_search_kernel_fragment(
      topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
    planner.add_sample_filter_device_function(filter_name);
    return planner.get_launcher();
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  single_cta_search::
    CagraSingleCtaSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
      planner(dataset_desc.metric,
              topk_by_bitonic_sort,
              bitonic_sort_and_merge_multi_warps,
              dataset_desc.team_size,
              dataset_desc.dataset_block_dim,
              dataset_desc.is_vpq,
              dataset_desc.pq_bits,
              dataset_desc.pq_len,
              persistent);

  planner.template add_setup_workspace_device_function<DataTag,
                                                       IndexTag,
                                                       DistTag,
                                                       QueryTag,
                                                       CodebookTag>(dataset_desc.metric,
                                                                    dataset_desc.team_size,
                                                                    dataset_desc.dataset_block_dim,
                                                                    dataset_desc.is_vpq,
                                                                    dataset_desc.pq_bits,
                                                                    dataset_desc.pq_len);
  planner.template add_compute_distance_device_function<DataTag,
                                                        IndexTag,
                                                        DistTag,
                                                        QueryTag,
                                                        CodebookTag>(dataset_desc.metric,
                                                                     dataset_desc.team_size,
                                                                     dataset_desc.dataset_block_dim,
                                                                     dataset_desc.is_vpq,
                                                                     dataset_desc.pq_bits,
                                                                     dataset_desc.pq_len);
  planner.add_search_kernel_fragment(
    topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
  planner.add_sample_filter_device_function(filter_name);
  return planner.get_launcher();
}

/// Build a JIT AlgorithmLauncher for multi-CTA CAGRA search.
template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> make_cagra_multi_cta_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const std::string& filter_name)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    multi_cta_search::
      CagraMultiCtaSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);

    planner.template add_setup_workspace_device_function<DataTag,
                                                         IndexTag,
                                                         DistTag,
                                                         QueryTag,
                                                         CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.template add_compute_distance_device_function<DataTag,
                                                          IndexTag,
                                                          DistTag,
                                                          QueryTag,
                                                          CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.add_search_multi_cta_kernel_fragment();
    planner.add_sample_filter_device_function(filter_name);
    return planner.get_launcher();
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    multi_cta_search::
      CagraMultiCtaSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);

    planner.template add_setup_workspace_device_function<DataTag,
                                                         IndexTag,
                                                         DistTag,
                                                         QueryTag,
                                                         CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.template add_compute_distance_device_function<DataTag,
                                                          IndexTag,
                                                          DistTag,
                                                          QueryTag,
                                                          CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.add_search_multi_cta_kernel_fragment();
    planner.add_sample_filter_device_function(filter_name);
    return planner.get_launcher();
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  multi_cta_search::
    CagraMultiCtaSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
      planner(dataset_desc.metric,
              dataset_desc.team_size,
              dataset_desc.dataset_block_dim,
              dataset_desc.is_vpq,
              dataset_desc.pq_bits,
              dataset_desc.pq_len);

  planner.template add_setup_workspace_device_function<DataTag,
                                                       IndexTag,
                                                       DistTag,
                                                       QueryTag,
                                                       CodebookTag>(dataset_desc.metric,
                                                                    dataset_desc.team_size,
                                                                    dataset_desc.dataset_block_dim,
                                                                    dataset_desc.is_vpq,
                                                                    dataset_desc.pq_bits,
                                                                    dataset_desc.pq_len);
  planner.template add_compute_distance_device_function<DataTag,
                                                        IndexTag,
                                                        DistTag,
                                                        QueryTag,
                                                        CodebookTag>(dataset_desc.metric,
                                                                     dataset_desc.team_size,
                                                                     dataset_desc.dataset_block_dim,
                                                                     dataset_desc.is_vpq,
                                                                     dataset_desc.pq_bits,
                                                                     dataset_desc.pq_len);
  planner.add_search_multi_cta_kernel_fragment();
  planner.add_sample_filter_device_function(filter_name);
  return planner.get_launcher();
}

/// Build a JIT AlgorithmLauncher for multi-kernel CAGRA helpers (random_pickup, compute_distance,
/// …).
template <typename DataT, typename IndexT, typename DistanceT, typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> make_cagra_multi_kernel_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const char* linked_kernel_name)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    multi_kernel_search::
      CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                linked_kernel_name,
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);
    planner.template add_setup_workspace_device_function<DataTag,
                                                         IndexTag,
                                                         DistTag,
                                                         QueryTag,
                                                         CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.template add_compute_distance_device_function<DataTag,
                                                          IndexTag,
                                                          DistTag,
                                                          QueryTag,
                                                          CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.add_linked_kernel(linked_kernel_name);
    return planner.get_launcher();
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    multi_kernel_search::
      CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
        planner(dataset_desc.metric,
                linked_kernel_name,
                dataset_desc.team_size,
                dataset_desc.dataset_block_dim,
                dataset_desc.is_vpq,
                dataset_desc.pq_bits,
                dataset_desc.pq_len);
    planner.template add_setup_workspace_device_function<DataTag,
                                                         IndexTag,
                                                         DistTag,
                                                         QueryTag,
                                                         CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.template add_compute_distance_device_function<DataTag,
                                                          IndexTag,
                                                          DistTag,
                                                          QueryTag,
                                                          CodebookTag>(
      dataset_desc.metric,
      dataset_desc.team_size,
      dataset_desc.dataset_block_dim,
      dataset_desc.is_vpq,
      dataset_desc.pq_bits,
      dataset_desc.pq_len);
    planner.add_linked_kernel(linked_kernel_name);
    return planner.get_launcher();
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  multi_kernel_search::
    CagraMultiKernelSearchPlanner<DataTag, IndexTag, DistTag, SourceTag, QueryTag, CodebookTag>
      planner(dataset_desc.metric,
              linked_kernel_name,
              dataset_desc.team_size,
              dataset_desc.dataset_block_dim,
              dataset_desc.is_vpq,
              dataset_desc.pq_bits,
              dataset_desc.pq_len);
  planner.template add_setup_workspace_device_function<DataTag,
                                                       IndexTag,
                                                       DistTag,
                                                       QueryTag,
                                                       CodebookTag>(dataset_desc.metric,
                                                                    dataset_desc.team_size,
                                                                    dataset_desc.dataset_block_dim,
                                                                    dataset_desc.is_vpq,
                                                                    dataset_desc.pq_bits,
                                                                    dataset_desc.pq_len);
  planner.template add_compute_distance_device_function<DataTag,
                                                        IndexTag,
                                                        DistTag,
                                                        QueryTag,
                                                        CodebookTag>(dataset_desc.metric,
                                                                     dataset_desc.team_size,
                                                                     dataset_desc.dataset_block_dim,
                                                                     dataset_desc.is_vpq,
                                                                     dataset_desc.pq_bits,
                                                                     dataset_desc.pq_len);
  planner.add_linked_kernel(linked_kernel_name);
  return planner.get_launcher();
}

}  // namespace cuvs::neighbors::cagra::detail
