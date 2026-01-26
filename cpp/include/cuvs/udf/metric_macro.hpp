/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "metric_interface.cuh"
#include "metric_source.hpp"
#include "point.cuh"

/**
 * @brief Define a custom distance metric with compile-time validation.
 *
 * This macro creates:
 * 1. A struct that inherits from metric_interface (compile-time validation)
 * 2. A function NAME_udf() that returns a metric_source for JIT compilation
 *
 * @param NAME The name of your metric (becomes struct name and function prefix)
 * @param BODY The body of operator()(AccT& acc, point_type x, point_type y)
 *
 * Available in BODY:
 *   acc    - Accumulated distance (AccT&, modify in-place)
 *   x, y   - Vector elements (point<T, AccT, Veclen>)
 *   T      - Data type (float, __half, int8_t, uint8_t)
 *   AccT   - Accumulator type
 *   Veclen - Vector length (compile-time constant)
 *
 * x and y provide:
 *   x.raw()     - Raw packed storage (for power users)
 *   x[i]        - Unpacked element access
 *   x.size()    - Number of elements (4 for packed int8, 1 for float)
 *   x.is_packed() - Whether data is packed (constexpr)
 *
 * Helper functions (Veclen deduced automatically!):
 *   cuvs::udf::squared_diff(x, y) - (x-y)² optimized for all types
 *   cuvs::udf::abs_diff(x, y)     - |x-y| optimized for all types
 *   cuvs::udf::dot_product(x, y)  - x·y optimized for all types
 *   cuvs::udf::product(x, y)      - element-wise product
 *
 * Example:
 *   CUVS_METRIC(my_l2, {
 *       acc += cuvs::udf::squared_diff(x, y);  // Just works for all types!
 *   })
 *
 *   CUVS_METRIC(my_chebyshev, {
 *       for (int i = 0; i < x.size(); ++i) {
 *           auto diff = (x[i] > y[i]) ? (x[i] - y[i]) : (y[i] - x[i]);
 *           if (diff > acc) acc = diff;
 *       }
 *   })
 */
#define CUVS_METRIC(NAME, BODY)                                                        \
  template <typename T, typename AccT, int Veclen>                                     \
  struct NAME : cuvs::udf::metric_interface<T, AccT, Veclen> {                         \
    using point_type = cuvs::udf::point<T, AccT, Veclen>;                              \
    __device__ void operator()(AccT& acc, point_type x, point_type y) override { BODY } \
  };                                                                                   \
                                                                                       \
  inline cuvs::udf::metric_source NAME##_udf()                                         \
  {                                                                                    \
    return cuvs::udf::metric_source{                                                   \
      .source = R"(                                                                    \
template <typename T, typename AccT, int Veclen>                                       \
struct )" #NAME R"( : cuvs::udf::metric_interface<T, AccT, Veclen> {                   \
    using point_type = cuvs::udf::point<T, AccT, Veclen>;                              \
    __device__ void operator()(AccT& acc, point_type x, point_type y) override         \
)" #BODY R"(                                                                           \
};                                                                                     \
)",                                                                                    \
      .struct_name = #NAME,                                                            \
      .headers     = {}};                                                              \
  }
