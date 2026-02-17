/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef CUVS_ENABLE_JIT_LTO
#error "shared_launcher_jit.hpp included but CUVS_ENABLE_JIT_LTO not defined!"
#endif

// Include tags header before any other includes that might open namespaces
#include <cuvs/detail/jit_lto/cagra/search_single_cta_tags.hpp>

#include "../../sample_filter.cuh"  // For none_sample_filter, bitset_filter

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
  if constexpr (std::is_same_v<T, float>) { return tag_f{}; }
  if constexpr (std::is_same_v<T, __half>) { return tag_h{}; }
  if constexpr (std::is_same_v<T, int8_t>) { return tag_sc{}; }
  if constexpr (std::is_same_v<T, uint8_t>) { return tag_uc{}; }
}

template <typename T>
constexpr auto get_index_type_tag()
{
  if constexpr (std::is_same_v<T, uint32_t>) { return tag_idx_ui{}; }
}

template <typename T>
constexpr auto get_distance_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return tag_dist_f{}; }
}

template <typename T>
constexpr auto get_source_index_type_tag()
{
  if constexpr (std::is_same_v<T, uint32_t>) { return tag_idx_ui{}; }
}

// Helper trait to detect if a type is a bitset_filter (regardless of template parameters)
template <typename T>
struct is_bitset_filter : std::false_type {};

template <typename bitset_t, typename index_t>
struct is_bitset_filter<cuvs::neighbors::filtering::bitset_filter<bitset_t, index_t>>
  : std::true_type {};

template <class SAMPLE_FILTER_T>
std::string get_sample_filter_name(bool debug_output = false)
{
  using namespace cuvs::neighbors::filtering;
  using DecayedFilter = std::decay_t<SAMPLE_FILTER_T>;

  if (debug_output) {
    std::cerr << "[JIT] get_sample_filter_name called" << std::endl;
    std::cerr << "[JIT] Type name: " << typeid(DecayedFilter).name() << std::endl;
  }

  // First check for none_sample_filter (the only unwrapped case)
  if constexpr (std::is_same_v<DecayedFilter, none_sample_filter>) {
    if (debug_output) { std::cerr << "[JIT] Returning: filter_none" << std::endl; }
    return "filter_none";
  }

  // All other filters are wrapped in CagraSampleFilterWithQueryIdOffset
  // Access the inner filter type via decltype
  if constexpr (requires { std::declval<DecayedFilter>().filter; }) {
    using InnerFilter = decltype(std::declval<DecayedFilter>().filter);
    if constexpr (is_bitset_filter<InnerFilter>::value ||
                  std::is_same_v<InnerFilter, bitset_filter<uint32_t, int64_t>> ||
                  std::is_same_v<InnerFilter, bitset_filter<uint32_t, uint32_t>>) {
      if (debug_output) {
        std::cerr << "[JIT] Returning: filter_bitset (via wrapped filter)" << std::endl;
      }
      return "filter_bitset";
    }
  }

  // Default to none filter for unknown types
  if (debug_output) { std::cerr << "[JIT] Returning: filter_none (default/unknown)" << std::endl; }
  return "filter_none";
}

}  // namespace cuvs::neighbors::cagra::detail
