/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace cuvs::neighbors::cagra::detail {

template <typename QUERY_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op(QUERY_T a, QUERY_T b)
{
  return -static_cast<DISTANCE_T>(a) * static_cast<DISTANCE_T>(b);
}

}  // namespace cuvs::neighbors::cagra::detail
