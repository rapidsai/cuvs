/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::cagra::detail {

template <typename QUERY_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op(QUERY_T a, QUERY_T b)
{
  const auto v = (a ^ b) & 0xffu;
  return __popc(v);
}

}  // namespace cuvs::neighbors::cagra::detail
