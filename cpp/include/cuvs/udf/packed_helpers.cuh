/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// =============================================================================
// DEPRECATED: Use point.cuh helpers instead!
//
// Old API:
//   acc += cuvs::udf::packed_sq_diff_u8(x, y);
//
// New API (recommended):
//   acc += cuvs::udf::squared_diff(x, y);  // Works for ALL types!
//
// The new helpers take point<T, AccT, Veclen> and deduce types automatically.
// =============================================================================

#include "point.cuh"

namespace cuvs::udf {

// Legacy helpers for raw packed values (use point-based helpers instead!)

/**
 * @brief [DEPRECATED] Use squared_diff(x, y) with point wrapper instead.
 */
__device__ __forceinline__ uint32_t packed_sq_diff_u8(uint32_t x, uint32_t y)
{
  auto diff = __vabsdiffu4(x, y);
  return __dp4a(diff, diff, 0u);
}

/**
 * @brief [DEPRECATED] Use squared_diff(x, y) with point wrapper instead.
 */
__device__ __forceinline__ int32_t packed_sq_diff_i8(int32_t x, int32_t y)
{
  auto diff = __vabsdiffs4(x, y);
  return __dp4a(diff, diff, 0);
}

/**
 * @brief [DEPRECATED] Use abs_diff(x, y) with point wrapper instead.
 */
__device__ __forceinline__ uint32_t packed_l1_u8(uint32_t x, uint32_t y)
{
  auto diff = __vabsdiffu4(x, y);
  return (diff & 0xFF) + ((diff >> 8) & 0xFF) + ((diff >> 16) & 0xFF) + ((diff >> 24) & 0xFF);
}

/**
 * @brief [DEPRECATED] Use abs_diff(x, y) with point wrapper instead.
 */
__device__ __forceinline__ int32_t packed_l1_i8(int32_t x, int32_t y)
{
  auto diff = __vabsdiffs4(x, y);
  uint32_t udiff = static_cast<uint32_t>(diff);
  return static_cast<int32_t>((udiff & 0xFF) + ((udiff >> 8) & 0xFF) + ((udiff >> 16) & 0xFF) +
                              ((udiff >> 24) & 0xFF));
}

/**
 * @brief [DEPRECATED] Use dot_product(x, y) with point wrapper instead.
 */
__device__ __forceinline__ uint32_t packed_dot_u8(uint32_t x, uint32_t y)
{
  return __dp4a(x, y, 0u);
}

/**
 * @brief [DEPRECATED] Use dot_product(x, y) with point wrapper instead.
 */
__device__ __forceinline__ int32_t packed_dot_i8(int32_t x, int32_t y)
{
  return __dp4a(x, y, 0);
}

}  // namespace cuvs::udf
