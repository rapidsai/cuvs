/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance.hpp"
#include "../shared_launcher_jit.hpp"
#include "search_multi_cta_planner.hpp"
#include "search_multi_kernel_planner.hpp"
#include "search_single_cta_planner.hpp"

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/distance/distance.hpp>

#include <memory>

namespace cuvs::neighbors::cagra::detail {

namespace cagra_jit_launcher_factory_detail {

template <typename DataTag,
          typename IndexTag,
          typename DistTag,
          typename SourceTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> build_single_cta_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  bool topk_by_bitonic_sort,
  bool bitonic_sort_and_merge_multi_warps,
  bool persistent)
{
  single_cta_search::CagraSingleCtaSearchPlanner<DataTag,
                                                 IndexTag,
                                                 DistTag,
                                                 SourceTag,
                                                 QueryTag,
                                                 CodebookTag,
                                                 SampleFilterJitTag>
    planner(dataset_desc.metric,
            topk_by_bitonic_sort,
            bitonic_sort_and_merge_multi_warps,
            dataset_desc.team_size,
            dataset_desc.dataset_block_dim,
            dataset_desc.is_vpq,
            dataset_desc.pq_bits,
            dataset_desc.pq_len,
            persistent);

  planner.add_setup_workspace_device_function(dataset_desc.metric,
                                              dataset_desc.team_size,
                                              dataset_desc.dataset_block_dim,
                                              dataset_desc.is_vpq,
                                              dataset_desc.pq_bits,
                                              dataset_desc.pq_len);
  planner.add_compute_distance_device_function(dataset_desc.metric,
                                               dataset_desc.team_size,
                                               dataset_desc.dataset_block_dim,
                                               dataset_desc.is_vpq,
                                               dataset_desc.pq_bits,
                                               dataset_desc.pq_len);
  planner.add_search_kernel_fragment(
    topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
  planner.add_sample_filter_device_function();
  return planner.get_launcher();
}

template <typename DataTag,
          typename IndexTag,
          typename DistTag,
          typename SourceTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> build_multi_cta_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc)
{
  multi_cta_search::CagraMultiCtaSearchPlanner<DataTag,
                                               IndexTag,
                                               DistTag,
                                               SourceTag,
                                               QueryTag,
                                               CodebookTag,
                                               SampleFilterJitTag>
    planner(dataset_desc.metric,
            dataset_desc.team_size,
            dataset_desc.dataset_block_dim,
            dataset_desc.is_vpq,
            dataset_desc.pq_bits,
            dataset_desc.pq_len);

  planner.add_setup_workspace_device_function(dataset_desc.metric,
                                              dataset_desc.team_size,
                                              dataset_desc.dataset_block_dim,
                                              dataset_desc.is_vpq,
                                              dataset_desc.pq_bits,
                                              dataset_desc.pq_len);
  planner.add_compute_distance_device_function(dataset_desc.metric,
                                               dataset_desc.team_size,
                                               dataset_desc.dataset_block_dim,
                                               dataset_desc.is_vpq,
                                               dataset_desc.pq_bits,
                                               dataset_desc.pq_len);
  planner.add_search_multi_cta_kernel_fragment();
  planner.add_sample_filter_device_function();
  return planner.get_launcher();
}

template <typename DataTag,
          typename IndexTag,
          typename DistTag,
          typename SourceTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
std::shared_ptr<AlgorithmLauncher> build_multi_kernel_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  const char* linked_kernel_name)
{
  multi_kernel_search::CagraMultiKernelSearchPlanner<DataTag,
                                                     IndexTag,
                                                     DistTag,
                                                     SourceTag,
                                                     QueryTag,
                                                     CodebookTag,
                                                     SampleFilterJitTag>
    planner(dataset_desc.metric,
            linked_kernel_name,
            dataset_desc.team_size,
            dataset_desc.dataset_block_dim,
            dataset_desc.is_vpq,
            dataset_desc.pq_bits,
            dataset_desc.pq_len);
  planner.add_setup_workspace_device_function(dataset_desc.metric,
                                              dataset_desc.team_size,
                                              dataset_desc.dataset_block_dim,
                                              dataset_desc.is_vpq,
                                              dataset_desc.pq_bits,
                                              dataset_desc.pq_len);
  planner.add_compute_distance_device_function(dataset_desc.metric,
                                               dataset_desc.team_size,
                                               dataset_desc.dataset_block_dim,
                                               dataset_desc.is_vpq,
                                               dataset_desc.pq_bits,
                                               dataset_desc.pq_len);
  planner.add_sample_filter_device_function();
  planner.add_linked_kernel(linked_kernel_name);
  return planner.get_launcher();
}

}  // namespace cagra_jit_launcher_factory_detail

/// Build a JIT AlgorithmLauncher for single-CTA CAGRA search (runtime VPQ / metric → tag
/// dispatch). `SampleFilterJitTag` is `cuvs::neighbors::detail::tag_filter_none`,
/// `tag_filter_bitset`, or use `sample_filter_jit_tag_t<SAMPLE_FILTER_T>`.
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag>
std::shared_ptr<AlgorithmLauncher> make_cagra_single_cta_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
  bool topk_by_bitonic_sort,
  bool bitonic_sort_and_merge_multi_warps,
  bool persistent)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    return cagra_jit_launcher_factory_detail::build_single_cta_launcher<DataTag,
                                                                        IndexTag,
                                                                        DistTag,
                                                                        SourceTag,
                                                                        QueryTag,
                                                                        CodebookTag,
                                                                        SampleFilterJitTag,
                                                                        DataT,
                                                                        IndexT,
                                                                        DistanceT,
                                                                        SourceIndexT>(
      dataset_desc, topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    return cagra_jit_launcher_factory_detail::build_single_cta_launcher<DataTag,
                                                                        IndexTag,
                                                                        DistTag,
                                                                        SourceTag,
                                                                        QueryTag,
                                                                        CodebookTag,
                                                                        SampleFilterJitTag,
                                                                        DataT,
                                                                        IndexT,
                                                                        DistanceT,
                                                                        SourceIndexT>(
      dataset_desc, topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  return cagra_jit_launcher_factory_detail::build_single_cta_launcher<DataTag,
                                                                      IndexTag,
                                                                      DistTag,
                                                                      SourceTag,
                                                                      QueryTag,
                                                                      CodebookTag,
                                                                      SampleFilterJitTag,
                                                                      DataT,
                                                                      IndexT,
                                                                      DistanceT,
                                                                      SourceIndexT>(
    dataset_desc, topk_by_bitonic_sort, bitonic_sort_and_merge_multi_warps, persistent);
}

/// Build a JIT AlgorithmLauncher for multi-CTA CAGRA search.
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag>
std::shared_ptr<AlgorithmLauncher> make_cagra_multi_cta_jit_launcher(
  const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc)
{
  using DataTag   = decltype(get_data_type_tag<DataT>());
  using IndexTag  = decltype(get_index_type_tag<IndexT>());
  using DistTag   = decltype(get_distance_type_tag<DistanceT>());
  using SourceTag = decltype(get_source_index_type_tag<SourceIndexT>());

  if (dataset_desc.is_vpq) {
    using QueryTag    = query_type_tag_vpq_t<DataTag>;
    using CodebookTag = codebook_tag_vpq_t;
    return cagra_jit_launcher_factory_detail::build_multi_cta_launcher<DataTag,
                                                                       IndexTag,
                                                                       DistTag,
                                                                       SourceTag,
                                                                       QueryTag,
                                                                       CodebookTag,
                                                                       SampleFilterJitTag,
                                                                       DataT,
                                                                       IndexT,
                                                                       DistanceT,
                                                                       SourceIndexT>(dataset_desc);
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    return cagra_jit_launcher_factory_detail::build_multi_cta_launcher<DataTag,
                                                                       IndexTag,
                                                                       DistTag,
                                                                       SourceTag,
                                                                       QueryTag,
                                                                       CodebookTag,
                                                                       SampleFilterJitTag,
                                                                       DataT,
                                                                       IndexT,
                                                                       DistanceT,
                                                                       SourceIndexT>(dataset_desc);
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  return cagra_jit_launcher_factory_detail::build_multi_cta_launcher<DataTag,
                                                                     IndexTag,
                                                                     DistTag,
                                                                     SourceTag,
                                                                     QueryTag,
                                                                     CodebookTag,
                                                                     SampleFilterJitTag,
                                                                     DataT,
                                                                     IndexT,
                                                                     DistanceT,
                                                                     SourceIndexT>(dataset_desc);
}

/// Build a JIT AlgorithmLauncher for multi-kernel CAGRA helpers (random_pickup, compute_distance,
/// …). Use `SampleFilterJitTag = tag_cagra_jit_sample_filter_link_absent` (default) when the kernel
/// does not link `sample_filter`; otherwise `sample_filter_jit_tag_t<SAMPLE_FILTER_T>` or a
/// `tag_filter_*` from `common_fragments.hpp`.
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          typename SampleFilterJitTag = tag_cagra_jit_sample_filter_link_absent>
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
    return cagra_jit_launcher_factory_detail::build_multi_kernel_launcher<DataTag,
                                                                          IndexTag,
                                                                          DistTag,
                                                                          SourceTag,
                                                                          QueryTag,
                                                                          CodebookTag,
                                                                          SampleFilterJitTag,
                                                                          DataT,
                                                                          IndexT,
                                                                          DistanceT,
                                                                          SourceIndexT>(
      dataset_desc, linked_kernel_name);
  }
  using CodebookTag = codebook_tag_standard_t;
  if (dataset_desc.metric == cuvs::distance::DistanceType::BitwiseHamming) {
    using QueryTag =
      query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;
    return cagra_jit_launcher_factory_detail::build_multi_kernel_launcher<DataTag,
                                                                          IndexTag,
                                                                          DistTag,
                                                                          SourceTag,
                                                                          QueryTag,
                                                                          CodebookTag,
                                                                          SampleFilterJitTag,
                                                                          DataT,
                                                                          IndexT,
                                                                          DistanceT,
                                                                          SourceIndexT>(
      dataset_desc, linked_kernel_name);
  }
  using QueryTag = query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
  return cagra_jit_launcher_factory_detail::build_multi_kernel_launcher<DataTag,
                                                                        IndexTag,
                                                                        DistTag,
                                                                        SourceTag,
                                                                        QueryTag,
                                                                        CodebookTag,
                                                                        SampleFilterJitTag,
                                                                        DataT,
                                                                        IndexT,
                                                                        DistanceT,
                                                                        SourceIndexT>(
    dataset_desc, linked_kernel_name);
}

}  // namespace cuvs::neighbors::cagra::detail
