/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../sample_filter.cuh"  // public filter types
#include "../../sample_filter_data.cuh"

#include <cstdint>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename SourceIndexT>
using cagra_filter_data_storage = ::cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>;

enum class cagra_filter_kind : std::uint32_t { none = 0, bitset = 1, udf = 2 };

/// Host/device payload for linked CAGRA sample filters plus query offset for wrapped filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  cagra_filter_data_storage<SourceIndexT> filter_data_storage{};
  void* filter_data{nullptr};
  cagra_filter_kind filter_kind{cagra_filter_kind::none};
  std::uint32_t query_id_offset{0};
};

template <typename T>
struct is_bitset_filter : std::false_type {};

template <typename bitset_t, typename index_t>
struct is_bitset_filter<::cuvs::neighbors::filtering::bitset_filter<bitset_t, index_t>>
  : std::true_type {};

template <typename T>
struct is_udf_filter : std::false_type {};

template <>
struct is_udf_filter<::cuvs::neighbors::filtering::udf_filter> : std::true_type {};

template <typename SourceIndexT, typename FilterT>
void fill_cagra_sample_filter(cagra_sample_filter<SourceIndexT>& out, const FilterT& filter)
{
  using DecayedFilter = std::decay_t<FilterT>;
  if constexpr (is_bitset_filter<DecayedFilter>::value) {
    const auto bitset_view = filter.view();
    out.filter_data_storage =
      cagra_filter_data_storage<SourceIndexT>{const_cast<std::uint32_t*>(bitset_view.data()),
                                              static_cast<SourceIndexT>(bitset_view.size()),
                                              static_cast<SourceIndexT>(
                                                bitset_view.get_original_nbits())};
    out.filter_kind = cagra_filter_kind::bitset;
  } else if constexpr (is_udf_filter<DecayedFilter>::value) {
    out.filter_data = filter.filter_data;
    out.filter_kind = cagra_filter_kind::udf;
  }
}

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
    fill_cagra_sample_filter(out, sample_filter.filter);
  } else {
    fill_cagra_sample_filter(out, sample_filter);
  }
  return out;
}

template <typename SampleFilterT>
const ::cuvs::neighbors::filtering::udf_filter* get_cagra_udf_filter(
  const SampleFilterT& sample_filter)
{
  using DecayedFilter = std::decay_t<SampleFilterT>;
  if constexpr (is_udf_filter<DecayedFilter>::value) {
    return &sample_filter;
  } else if constexpr (requires { sample_filter.filter; }) {
    return get_cagra_udf_filter(sample_filter.filter);
  } else {
    return nullptr;
  }
}

template <typename SourceIndexT>
__device__ __forceinline__ void* get_cagra_sample_filter_data(
  cagra_sample_filter<SourceIndexT>& payload)
{
  if (payload.filter_kind == cagra_filter_kind::udf) { return payload.filter_data; }
  if (payload.filter_kind == cagra_filter_kind::bitset) {
    // The payload is passed by value to kernels; take the embedded storage address on device.
    return &payload.filter_data_storage;
  }
  return nullptr;
}

}  // namespace cuvs::neighbors::cagra::detail
