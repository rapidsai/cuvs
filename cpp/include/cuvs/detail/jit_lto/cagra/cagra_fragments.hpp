/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/detail/jit_lto/common_fragments.hpp>

namespace cuvs::neighbors::cagra::detail {

struct tag_dist_f {};
struct tag_metric_l2 {};
struct tag_metric_inner_product {};
struct tag_metric_cosine {};
struct tag_metric_hamming {};
struct tag_team_8 {};
struct tag_team_16 {};
struct tag_team_32 {};
struct tag_dim_128 {};
struct tag_dim_256 {};
struct tag_dim_512 {};
struct tag_pq_bits_0 {};
struct tag_pq_bits_8 {};
struct tag_pq_len_0 {};
struct tag_pq_len_2 {};
struct tag_pq_len_4 {};
struct tag_codebook_none {};
struct tag_codebook_half {};
struct tag_metric_l1 {};
struct tag_norm_noop {};
struct tag_norm_cosine {};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          typename CodebookTag,
          typename TeamTag,
          typename BlockDimTag,
          typename PqBitsTag,
          typename PqLenTag>
struct fragment_tag_setup_workspace {};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          typename CodebookTag,
          typename TeamTag,
          typename BlockDimTag,
          typename PqBitsTag,
          typename PqLenTag>
struct fragment_tag_compute_distance {};

template <typename QueryTag, typename DistanceTag, typename MetricTag>
struct fragment_tag_dist_op {};

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename QueryTag,
          typename TeamTag,
          typename BlockDimTag,
          typename NormTag>
struct fragment_tag_apply_normalization_standard {};

template <typename DataTag,
          typename SourceIndexTag,
          typename IndexTag,
          typename DistanceTag,
          bool TopkByBitonicSort,
          bool BitonicSortAndMergeMultiWarps>
struct fragment_tag_search_single_cta {};

template <typename DataTag,
          typename SourceIndexTag,
          typename IndexTag,
          typename DistanceTag,
          bool TopkByBitonicSort,
          bool BitonicSortAndMergeMultiWarps>
struct fragment_tag_search_single_cta_p {};

template <typename DataTag, typename SourceIndexTag, typename IndexTag, typename DistanceTag>
struct fragment_tag_search_multi_cta {};

template <typename DataTag, typename IndexTag, typename DistanceTag>
struct fragment_tag_random_pickup {};

template <typename DataTag, typename IndexTag, typename DistanceTag, typename SourceIndexTag>
struct fragment_tag_compute_distance_to_child_nodes {};

template <typename IndexTag, typename DistanceTag, typename SourceIndexTag>
struct fragment_tag_apply_filter_kernel {};

template <typename BitsetTag, typename SourceIndexTag, typename FilterTag>
struct fragment_tag_sample_filter {};

}  // namespace cuvs::neighbors::cagra::detail
