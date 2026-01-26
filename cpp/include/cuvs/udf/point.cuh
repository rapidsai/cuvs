/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace cuvs::udf {

/**
 * @brief Wrapper for vector elements that provides both packed and unpacked access.
 *
 * For float/half: trivial wrapper around scalar values
 * For int8/uint8 with Veclen > 1: wraps packed bytes in a 32-bit word
 *
 * @tparam T Data type (float, __half, int8_t, uint8_t)
 * @tparam AccT Storage/accumulator type (float, __half, int32_t, uint32_t)
 * @tparam Veclen Vector length (1, 2, 4, 8, 16)
 *
 * Usage:
 *   // Helpers deduce Veclen automatically:
 *   acc += cuvs::udf::squared_diff(x, y);  // No template args!
 *
 *   // Array access for custom logic (slower but flexible):
 *   for (int i = 0; i < x.size(); ++i) {
 *       acc += x[i] * y[i];
 *   }
 *
 *   // Query packing:
 *   if constexpr (decltype(x)::is_packed()) { ... }
 */
template <typename T, typename AccT, int Veclen>
struct point {
  using element_type = T;
  using storage_type = AccT;
  static constexpr int veclen = Veclen;

  storage_type data_;

  // ============================================================
  // Constructors
  // ============================================================

  __device__ __host__ point() = default;
  __device__ __host__ explicit point(storage_type d) : data_(d) {}

  // ============================================================
  // Raw access (for power users who need intrinsics)
  // ============================================================

  __device__ __forceinline__ storage_type raw() const { return data_; }
  __device__ __forceinline__ storage_type& raw() { return data_; }

  // ============================================================
  // Compile-time queries
  // ============================================================

  __device__ __host__ static constexpr int size()
  {
    // For packed int8/uint8: 4 elements per storage word
    if constexpr ((std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) && Veclen > 1) {
      return 4;
    } else {
      return 1;
    }
  }

  __device__ __host__ static constexpr bool is_packed()
  {
    return (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) && Veclen > 1;
  }

  // ============================================================
  // Element access (unpacks for int8/uint8)
  // ============================================================

  __device__ __forceinline__ T operator[](int i) const
  {
    if constexpr (std::is_same_v<T, int8_t> && Veclen > 1) {
      // Extract signed byte i from packed int32_t
      return static_cast<int8_t>((data_ >> (i * 8)) & 0xFF);
    } else if constexpr (std::is_same_v<T, uint8_t> && Veclen > 1) {
      // Extract unsigned byte i from packed uint32_t
      return static_cast<uint8_t>((data_ >> (i * 8)) & 0xFF);
    } else {
      // Scalar types: only one element
      (void)i;  // Unused
      return static_cast<T>(data_);
    }
  }
};

// ============================================================
// Helper Operations - Deduce Veclen from point type!
// ============================================================

/**
 * @brief Squared difference: (x - y)²
 *
 * Optimized for packed int8/uint8, falls back to scalar for float/half.
 */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT squared_diff(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr (std::is_same_v<T, uint8_t> && V > 1) {
    // SIMD: 4 packed unsigned bytes
    auto diff = __vabsdiffu4(x.raw(), y.raw());
    return __dp4a(diff, diff, AccT{0});
  } else if constexpr (std::is_same_v<T, int8_t> && V > 1) {
    // SIMD: 4 packed signed bytes
    auto diff = __vabsdiffs4(x.raw(), y.raw());
    return __dp4a(diff, diff, static_cast<uint32_t>(0));
  } else {
    // Scalar: float, half, or byte with Veclen==1
    auto diff = x.raw() - y.raw();
    return diff * diff;
  }
}

/**
 * @brief Absolute difference: |x - y|
 *
 * For packed types, returns sum of absolute differences.
 */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT abs_diff(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr (std::is_same_v<T, uint8_t> && V > 1) {
    // SIMD: sum of 4 unsigned absolute differences
    auto diff = __vabsdiffu4(x.raw(), y.raw());
    // Sum the 4 bytes
    return ((diff >> 0) & 0xFF) + ((diff >> 8) & 0xFF) + ((diff >> 16) & 0xFF) +
           ((diff >> 24) & 0xFF);
  } else if constexpr (std::is_same_v<T, int8_t> && V > 1) {
    // SIMD: sum of 4 signed absolute differences
    auto diff = __vabsdiffs4(x.raw(), y.raw());
    return ((diff >> 0) & 0xFF) + ((diff >> 8) & 0xFF) + ((diff >> 16) & 0xFF) +
           ((diff >> 24) & 0xFF);
  } else {
    // Scalar
    auto a = x.raw();
    auto b = y.raw();
    return (a > b) ? (a - b) : (b - a);
  }
}

/**
 * @brief Dot product: x · y
 *
 * For packed types, computes sum of element-wise products.
 */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT dot_product(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    // SIMD: dp4a computes dot product of 4 packed bytes
    return __dp4a(x.raw(), y.raw(), AccT{0});
  } else {
    // Scalar
    return x.raw() * y.raw();
  }
}

/**
 * @brief Element-wise product: x * y
 *
 * For packed types, returns sum of element-wise products (same as dot_product).
 */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT product(point<T, AccT, V> x, point<T, AccT, V> y)
{
  return dot_product(x, y);
}

/**
 * @brief Element-wise sum: x + y
 */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT sum(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    // Sum all unpacked elements
    AccT result = 0;
    for (int i = 0; i < x.size(); ++i) {
      result += static_cast<AccT>(x[i]) + static_cast<AccT>(y[i]);
    }
    return result;
  } else {
    return x.raw() + y.raw();
  }
}

/**
 * @brief Maximum element: max(x, y)
 *
 * For packed types, returns max across all element pairs.
 */
template <typename T, typename AccT, int V>
__device__ __forceinline__ AccT max_elem(point<T, AccT, V> x, point<T, AccT, V> y)
{
  if constexpr ((std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) && V > 1) {
    AccT result = 0;
    for (int i = 0; i < x.size(); ++i) {
      auto xi  = static_cast<AccT>(x[i]);
      auto yi  = static_cast<AccT>(y[i]);
      auto val = (xi > yi) ? xi : yi;
      if (val > result) result = val;
    }
    return result;
  } else {
    auto a = x.raw();
    auto b = y.raw();
    return (a > b) ? a : b;
  }
}

}  // namespace cuvs::udf
