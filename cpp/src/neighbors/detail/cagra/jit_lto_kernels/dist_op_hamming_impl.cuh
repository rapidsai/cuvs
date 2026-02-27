/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

// dist_op fragment for BitwiseHamming metric
// QueryT is uint8_t for BitwiseHamming
template <typename QUERY_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op(QUERY_T a, QUERY_T b)
{
  // mask the result of xor for the integer promotion
  const auto v = (a ^ b) & 0xffu;
  return __popc(v);
}

}  // namespace cuvs::neighbors::cagra::detail
