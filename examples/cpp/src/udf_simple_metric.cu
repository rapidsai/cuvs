/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file udf_simple_metric.cu
 * @brief Simple example: Custom "Over 9000" L1 distance metric
 *
 * Shows the minimal code needed to define and use a custom metric.
 */

#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/udf/metric_macro.hpp>
#include <raft/core/device_resources.hpp>

#include <iostream>

// ============================================================
// Define your metric - ONE LINE (plus the body)!
// ============================================================
//
// The CUVS_METRIC macro:
// 1. Creates the actual struct (compiled, validated with `override`)
// 2. Generates awesome_over_9000_udf() function returning source string
//
// Available variables in body:
//   acc  - accumulated distance (AccT&, modify in-place)
//   x, y - vector elements (point<T, AccT, Veclen>)
//
// x and y provide:
//   x.raw()  - raw packed storage (power users)
//   x[i]     - element access (unpacked)
//   x.size() - number of elements
//
// Helpers (Veclen deduced automatically!):
//   cuvs::udf::squared_diff(x, y) - optimal for all types
//   cuvs::udf::abs_diff(x, y)     - optimal for all types
//   cuvs::udf::dot_product(x, y)  - optimal for all types

CUVS_METRIC(awesome_over_9000, {
  // IT'S OVER 9000!!!!
  // Works for ALL types - float, half, int8, uint8!
  auto diff = cuvs::udf::abs_diff(x, y);
  acc += diff * AccT{9001};
})

// That's it! The macro handles:
// - Struct definition with proper inheritance
// - operator() signature with `override` for validation
// - Source string generation for JIT
// - point<T,AccT,Veclen> wrapping for clean API

int main()
{
  std::cout << "=== cuVS UDF Simple Example ===\n\n";

  // ============================================================
  // Use in search
  // ============================================================

  // raft::device_resources res;
  // auto index = cuvs::neighbors::ivf_flat::deserialize(res, "index.bin");

  // cuvs::neighbors::ivf_flat::search_params params;
  // params.n_probes = 50;

  // Use the auto-generated _udf() function!
  // params.udf.metric = awesome_over_9000_udf();

  // cuvs::neighbors::ivf_flat::search(res, params, index, queries, neighbors, distances);

  // ============================================================
  // What happens under the hood
  // ============================================================

  std::cout << "User writes:\n";
  std::cout << "  CUVS_METRIC(awesome_over_9000, {\n";
  std::cout << "      auto diff = cuvs::udf::abs_diff(x, y);\n";
  std::cout << "      acc += diff * AccT{9001};\n";
  std::cout << "  })\n\n";

  std::cout << "x and y are point<T, AccT, Veclen> which provides:\n";
  std::cout << "  - x.raw()     : packed storage for intrinsics\n";
  std::cout << "  - x[i]        : unpacked element access\n";
  std::cout << "  - x.size()    : number of elements (4 for packed int8, 1 for float)\n";
  std::cout << "  - x.is_packed(): whether data is packed\n\n";

  std::cout << "Helper functions deduce Veclen automatically:\n";
  std::cout << "  cuvs::udf::squared_diff(x, y)  // No template args!\n";
  std::cout << "  cuvs::udf::abs_diff(x, y)\n";
  std::cout << "  cuvs::udf::dot_product(x, y)\n\n";

  std::cout << "At runtime, cuVS wraps raw values in point<T, AccT, Veclen>\n";
  std::cout << "and calls your metric with the wrapped arguments.\n";

  return 0;
}
