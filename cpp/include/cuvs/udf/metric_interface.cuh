/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "point.cuh"

namespace cuvs::udf {

/**
 * @brief Base interface for custom distance metrics.
 *
 * Inherit from this interface to get compile-time enforcement of the
 * correct operator() signature via the `override` keyword.
 *
 * If you forget to implement operator() or use the wrong signature,
 * you'll get a clear compile error: "does not override any member function"
 *
 * @tparam T Data type (float, __half, int8_t, uint8_t)
 * @tparam AccT Accumulator type (float, __half, int32_t, uint32_t)
 * @tparam Veclen Vector length (handled by cuVS internally)
 *
 * @note x and y are point<T, AccT, Veclen> which provides:
 *       - .raw()     : packed storage for power users
 *       - operator[] : unpacked element access
 *       - ::veclen   : compile-time Veclen
 *       - ::is_packed() : whether data is packed
 */
template <typename T, typename AccT, int Veclen = 1>
struct metric_interface {
  using point_type = point<T, AccT, Veclen>;

  /**
   * @brief Compute distance contribution for one element pair.
   *
   * @param[in,out] acc Accumulated distance value
   * @param[in] x Query vector element (point wrapper)
   * @param[in] y Database vector element (point wrapper)
   *
   * Example:
   *   // Simple - use helpers (recommended):
   *   acc += squared_diff(x, y);
   *
   *   // Array access for custom logic:
   *   for (int i = 0; i < x.size(); ++i) {
   *       acc += x[i] * y[i];
   *   }
   *
   *   // Power user - raw access:
   *   if constexpr (point_type::is_packed()) {
   *       acc = __dp4a(x.raw(), y.raw(), acc);
   *   }
   */
  virtual __device__ void operator()(AccT& acc, point_type x, point_type y) = 0;

  virtual __device__ ~metric_interface() = default;
};

}  // namespace cuvs::udf
