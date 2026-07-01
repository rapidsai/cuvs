/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../sample_filter.cuh"  // bitset_filter, none_sample_filter
#include "../../sample_filter_data.cuh"

#include <cstdint>
#include <type_traits>

// The single-partition sample-filter payload (cagra_sample_filter, is_bitset_filter,
// extract_cagra_sample_filter) now lives in cagra_filter_payload.{hpp,cuh}. This header retains
// only the multi-partition-specific helpers, which have no upstream equivalent.
namespace cuvs::neighbors::cagra::detail {

template <typename SourceIndexT>
using cagra_bitset = cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>;

template <typename SourceIndexT>
using mp_cagra_bitset = cuvs::neighbors::detail::mp_bitset_filter_data_t<SourceIndexT>;

/// Multi-partition bitset payload for kernels plus query offset for wrapped CAGRA filters.
template <typename SourceIndexT>
struct mp_cagra_sample_filter {
  mp_cagra_bitset<SourceIndexT> bitset{};
  std::uint32_t query_id_offset{0};
};

template <typename T>
struct is_mp_bitset_filter : std::false_type {};

template <typename bitset_t, typename index_t>
struct is_mp_bitset_filter<
  cuvs::neighbors::filtering::multi_partition_bitset_filter<bitset_t, index_t>> : std::true_type {};

/// Host: fill @ref mp_cagra_sample_filter from a multi-partition CAGRA filter (mp JIT launchers).
template <typename SourceIndexT, typename SampleFilterT>
mp_cagra_sample_filter<SourceIndexT> extract_cagra_mp_sample_filter(
  const SampleFilterT& sample_filter)
{
  mp_cagra_sample_filter<SourceIndexT> out;
  if constexpr (is_mp_bitset_filter<std::decay_t<SampleFilterT>>::value) {
    const auto& combined         = sample_filter.combined_bitset_;
    out.bitset.bitset_ptr        = const_cast<std::uint32_t*>(combined.data());
    out.bitset.bitset_len        = static_cast<SourceIndexT>(combined.size());
    out.bitset.original_nbits    = static_cast<SourceIndexT>(combined.get_original_nbits());
    out.bitset.partition_offsets = sample_filter.partition_offsets_;
  }
  return out;
}

}  // namespace cuvs::neighbors::cagra::detail
