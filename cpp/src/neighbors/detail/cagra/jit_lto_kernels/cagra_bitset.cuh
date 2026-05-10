/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../sample_filter.cuh"  // bitset_filter, none_sample_filter
#include "../../sample_filter_data.cuh"

#include <cstdint>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename SourceIndexT>
using cagra_bitset = cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>;

/// Host: bitset payload for kernels plus query offset for wrapped CAGRA filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  cagra_bitset<SourceIndexT> bitset{};
  std::uint32_t query_id_offset{0};
};

template <typename T>
struct is_bitset_filter : std::false_type {};

template <typename bitset_t, typename index_t>
struct is_bitset_filter<cuvs::neighbors::filtering::bitset_filter<bitset_t, index_t>>
  : std::true_type {};

/// Host: fill @ref cagra_sample_filter from a CAGRA filter object (used by JIT LTO launchers).
template <typename SourceIndexT, typename SampleFilterT>
cagra_sample_filter<SourceIndexT> extract_cagra_sample_filter(const SampleFilterT& sample_filter)
{
  cagra_sample_filter<SourceIndexT> out;
  if constexpr (requires {
                  sample_filter.filter;
                  sample_filter.offset;
                }) {
    out.query_id_offset = sample_filter.offset;
    using InnerFilter   = decltype(sample_filter.filter);
    if constexpr (is_bitset_filter<InnerFilter>::value) {
      const auto bitset_view    = sample_filter.filter.view();
      out.bitset.bitset_ptr     = const_cast<std::uint32_t*>(bitset_view.data());
      out.bitset.bitset_len     = static_cast<SourceIndexT>(bitset_view.size());
      out.bitset.original_nbits = static_cast<SourceIndexT>(bitset_view.get_original_nbits());
    }
  }
  return out;
}

}  // namespace cuvs::neighbors::cagra::detail
