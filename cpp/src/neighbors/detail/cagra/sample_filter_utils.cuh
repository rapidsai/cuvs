/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sample_filter.cuh"

#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::cagra::detail {

template <class CagraSampleFilterT>
struct CagraSampleFilterWithQueryIdOffset {
  const uint32_t offset;
  CagraSampleFilterT filter;

  CagraSampleFilterWithQueryIdOffset(const uint32_t offset, const CagraSampleFilterT filter)
    : offset(offset), filter(filter)
  {
  }

  _RAFT_DEVICE auto operator()(const uint32_t query_id, const uint32_t sample_id)
  {
    return filter(query_id + offset, sample_id);
  }
};

template <class CagraSampleFilterT>
struct CagraSampleFilterT_Selector {
  using type = CagraSampleFilterWithQueryIdOffset<CagraSampleFilterT>;
};
template <>
struct CagraSampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter> {
  using type = cuvs::neighbors::filtering::none_sample_filter;
};

// A helper function to set a query id offset
template <class CagraSampleFilterT>
inline auto set_offset(CagraSampleFilterT filter, const uint32_t offset) ->
  typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type
{
  typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type new_filter(offset, filter);
  return new_filter;
}
template <>
inline auto set_offset<cuvs::neighbors::filtering::none_sample_filter>(
  cuvs::neighbors::filtering::none_sample_filter filter, const uint32_t) ->
  typename CagraSampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter>::type
{
  return filter;
}
}  // namespace cuvs::neighbors::cagra::detail
