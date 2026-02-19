/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::cagra::detail {

// dist_op fragment for L2Expanded metric
template <typename DATA_T, typename DISTANCE_T>
__device__ DISTANCE_T dist_op(DATA_T a, DATA_T b)
{
  DISTANCE_T diff = a - b;
  return diff * diff;
}

}  // namespace cuvs::neighbors::cagra::detail
