/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Include tags header before any other includes that might open namespaces
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>

#include "../../sample_filter.cuh"           // For none_sample_filter, bitset_filter
#include "jit_lto_kernels/cagra_bitset.cuh"  // is_bitset_filter, cagra_bitset, cagra_sample_filter, extract

#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace cuvs::neighbors::cagra::detail {

// Helper functions to get tags for JIT LTO
template <typename T>
constexpr auto get_data_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return cuvs::neighbors::detail::tag_f{}; }
  if constexpr (std::is_same_v<T, __half>) { return cuvs::neighbors::detail::tag_h{}; }
  if constexpr (std::is_same_v<T, int8_t>) { return cuvs::neighbors::detail::tag_i8{}; }
  if constexpr (std::is_same_v<T, uint8_t>) { return cuvs::neighbors::detail::tag_u8{}; }
}

template <typename T>
constexpr auto get_index_type_tag()
{
  if constexpr (std::is_same_v<T, uint32_t>) { return cuvs::neighbors::detail::tag_index_u32{}; }
}

template <typename T>
constexpr auto get_distance_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return cuvs::neighbors::cagra::detail::tag_dist_f{}; }
}

template <typename T>
constexpr auto get_source_index_type_tag()
{
  if constexpr (std::is_same_v<T, uint32_t>) { return cuvs::neighbors::detail::tag_index_u32{}; }
  if constexpr (std::is_same_v<T, int64_t>) { return cuvs::neighbors::detail::tag_index_i64{}; }
}

template <typename DataTag, cuvs::distance::DistanceType metric>
struct query_type_tag_standard {
  using type = std::conditional_t<metric == cuvs::distance::DistanceType::BitwiseHamming &&
                                    std::is_same_v<DataTag, cuvs::neighbors::detail::tag_u8>,
                                  cuvs::neighbors::detail::tag_u8,
                                  cuvs::neighbors::detail::tag_f>;
};

template <typename DataTag, cuvs::distance::DistanceType metric>
using query_type_tag_standard_t = typename query_type_tag_standard<DataTag, metric>::type;

template <typename DataTag>
using query_type_tag_vpq_t = cuvs::neighbors::detail::tag_h;

template <typename DataTag>
using query_type_tag_standard_l2_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::L2Expanded>;
template <typename DataTag>
using query_type_tag_standard_inner_product_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::InnerProduct>;
template <typename DataTag>
using query_type_tag_standard_cosine_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::CosineExpanded>;
template <typename DataTag>
using query_type_tag_standard_hamming_t =
  query_type_tag_standard_t<DataTag, cuvs::distance::DistanceType::BitwiseHamming>;

using codebook_tag_vpq_t      = cuvs::neighbors::cagra::detail::tag_codebook_half;
using codebook_tag_standard_t = cuvs::neighbors::cagra::detail::tag_codebook_none;

// Dependent false for static_assert in get_sample_filter_name (CAGRA JIT).
template <typename T>
inline constexpr bool cagra_jit_get_sample_filter_name_type_always_false = false;

template <class SAMPLE_FILTER_T>
std::string get_sample_filter_name()
{
  using namespace cuvs::neighbors::filtering;
  using DecayedFilter = std::decay_t<SAMPLE_FILTER_T>;

  if constexpr (std::is_same_v<DecayedFilter, none_sample_filter>) {
    return "filter_none_source_index_ui";
  } else if constexpr (requires { std::declval<DecayedFilter>().filter; }) {
    using InnerFilter = decltype(std::declval<DecayedFilter>().filter);
    if constexpr (is_bitset_filter<InnerFilter>::value ||
                  std::is_same_v<InnerFilter, bitset_filter<uint32_t, int64_t>> ||
                  std::is_same_v<InnerFilter, bitset_filter<uint32_t, uint32_t>>) {
      return "filter_bitset_source_index_ui";
    } else {
      static_assert(
        cagra_jit_get_sample_filter_name_type_always_false<DecayedFilter>,
        "CAGRA JIT: get_sample_filter_name does not know how to link this filter. "
        "CagraSampleFilterWithQueryIdOffset<Inner> requires Inner of type bitset_filter<bitset, "
        "SourceIndexT> (see cagra_bitset.cuh is_bitset_filter and sample_filter_utils.cuh). "
        "For a new filter kind, add a get_sample_filter_name() branch. "
        "(SAMPLE_FILTER_T in error = DecayedFilter, check InnerFilter in compiler output.)");
    }
  } else {
    static_assert(
      cagra_jit_get_sample_filter_name_type_always_false<DecayedFilter>,
      "CAGRA JIT: get_sample_filter_name: SAMPLE_FILTER_T must be cuvs::neighbors::filtering::"
      "none_sample_filter, or cuvs::neighbors::cagra::detail::CagraSampleFilterWithQueryIdOffset<"
      "bitset_filter<bitset, SourceIndexT>>. Unknown wrapper type. "
      "(SAMPLE_FILTER_T in error = DecayedFilter; add a branch in get_sample_filter_name().)");
  }
}

}  // namespace cuvs::neighbors::cagra::detail
