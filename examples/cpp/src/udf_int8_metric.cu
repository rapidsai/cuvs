/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file udf_int8_metric.cu
 * @brief Example: int8/uint8 metrics - now EASY with point wrapper!
 *
 * The point<T, AccT, Veclen> wrapper makes int8/uint8 metrics trivial.
 * No more manual intrinsics or if constexpr branches!
 */

#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/udf/metric_macro.hpp>
#include <raft/core/device_resources.hpp>

#include <iostream>

// ============================================================
// Universal L2 Distance - ONE LINE!
// ============================================================
//
// The helper squared_diff() handles ALL types optimally:
// - float/half: simple scalar math
// - int8/uint8: SIMD intrinsics (__vabsdiffs4, __dp4a)
//
// You don't need to know about packed types or intrinsics!

CUVS_METRIC(universal_l2, {
  acc += cuvs::udf::squared_diff(x, y);  // Just works for everything!
})

// ============================================================
// Universal L1 Distance - ONE LINE!
// ============================================================

CUVS_METRIC(universal_l1, {
  acc += cuvs::udf::abs_diff(x, y);  // Just works for everything!
})

// ============================================================
// Universal Dot Product - ONE LINE!
// ============================================================

CUVS_METRIC(universal_dot, {
  acc += cuvs::udf::dot_product(x, y);  // Just works for everything!
})

// ============================================================
// Custom logic using element access
// ============================================================
//
// For custom logic, use x[i] and y[i] to access individual elements.
// The point wrapper handles unpacking automatically.

CUVS_METRIC(custom_weighted_l2, {
  // Access individual elements - works for all types
  for (int i = 0; i < x.size(); ++i) {
    auto diff   = x[i] - y[i];
    auto weight = AccT{1} + AccT{i};  // Custom per-dimension weight
    acc += weight * diff * diff;
  }
})

// ============================================================
// Power user: raw access with intrinsics
// ============================================================
//
// For maximum performance, you can still use raw() and intrinsics.
// But now you don't HAVE to!

CUVS_METRIC(power_user_l2, {
  if constexpr (decltype(x)::is_packed()) {
    // SIMD path - use intrinsics directly
    auto diff = __vabsdiffs4(x.raw(), y.raw());
    acc       = __dp4a(diff, diff, acc);
  } else {
    // Scalar path
    auto diff = x.raw() - y.raw();
    acc += diff * diff;
  }
})

int main()
{
  std::cout << "=== cuVS UDF int8/uint8 Metrics - Now Easy! ===\n\n";

  std::cout << "OLD WAY (manual intrinsics):\n";
  std::cout << "  if constexpr (std::is_same_v<T, int8_t> && Veclen > 1) {\n";
  std::cout << "      auto diff = __vabsdiffs4(x, y);  // Must know this!\n";
  std::cout << "      acc = raft::dp4a(diff, diff, acc);  // And this!\n";
  std::cout << "  } else { ... }\n\n";

  std::cout << "NEW WAY (with point wrapper):\n";
  std::cout << "  acc += cuvs::udf::squared_diff(x, y);  // Just works!\n\n";

  std::cout << "Available helpers (auto-deduce Veclen):\n";
  std::cout << "  squared_diff(x, y)  - (x-y)² optimized for all types\n";
  std::cout << "  abs_diff(x, y)      - |x-y| optimized for all types\n";
  std::cout << "  dot_product(x, y)   - x·y optimized for all types\n";
  std::cout << "  product(x, y)       - element-wise product\n";
  std::cout << "  sum(x, y)           - element-wise sum\n";
  std::cout << "  max_elem(x, y)      - maximum element\n\n";

  std::cout << "For custom logic, use element access:\n";
  std::cout << "  for (int i = 0; i < x.size(); ++i) {\n";
  std::cout << "      acc += custom_weight[i] * (x[i] - y[i]);\n";
  std::cout << "  }\n\n";

  std::cout << "Type info available at compile time:\n";
  std::cout << "  x.size()       - 4 for packed int8/uint8, 1 for float\n";
  std::cout << "  x.is_packed()  - true for int8/uint8 with Veclen > 1\n";
  std::cout << "  x.raw()        - raw storage for power users\n";

  return 0;
}
