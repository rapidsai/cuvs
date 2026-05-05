/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_sq::detail {

// Per-row distance finalize. Called once per scanned row.
//
// L2/IP:  return dist
// Cosine: denom = query_norm * sqrtf(v_norm_sq);
//         return denom > 0 ? 1 - dist/denom : 0

__device__ float finalize_distance_l2_impl(float dist,
                                           float /* v_norm_sq */,
                                           float /* query_norm */)
{
  return dist;
}

__device__ float finalize_distance_ip_impl(float dist,
                                           float /* v_norm_sq */,
                                           float /* query_norm */)
{
  return dist;
}

__device__ float finalize_distance_cosine_impl(float dist, float v_norm_sq, float query_norm)
{
  float denom = query_norm * sqrtf(v_norm_sq);
  return (denom > 0.0f) ? 1.0f - dist / denom : 0.0f;
}

}  // namespace cuvs::neighbors::ivf_sq::detail
