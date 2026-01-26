/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file udf_chebyshev_metric.cu
 * @brief Example: Chebyshev (L∞) distance metric
 *
 * Chebyshev distance = max absolute difference across dimensions:
 *   d(x, y) = max_i |x_i - y_i|
 *
 * This example shows how to use element access for custom reduction logic.
 */

#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/udf/metric_macro.hpp>
#include <raft/core/device_resources.hpp>

#include <iostream>

// ============================================================
// Chebyshev (L∞) Distance
// ============================================================
//
// Unlike L2 (sum of squares) or L1 (sum of abs), Chebyshev
// tracks the MAXIMUM absolute difference seen so far.
//
// Uses element access x[i], y[i] for custom reduction.

CUVS_METRIC(chebyshev_distance, {
  for (int i = 0; i < x.size(); ++i) {
    auto xi   = x[i];
    auto yi   = y[i];
    auto diff = (xi > yi) ? (xi - yi) : (yi - xi);
    if (diff > acc) { acc = static_cast<AccT>(diff); }
  }
})

// ============================================================
// Weighted L1 Distance - using helper
// ============================================================

CUVS_METRIC(weighted_l1, {
  acc += cuvs::udf::abs_diff(x, y) * AccT{2.5};  // Custom weight
})

// ============================================================
// Squared L2 (Euclidean) Distance - using helper
// ============================================================

CUVS_METRIC(squared_l2, { acc += cuvs::udf::squared_diff(x, y); })

// ============================================================
// Minkowski Distance (p=3) - using element access
// ============================================================

CUVS_METRIC(minkowski_p3, {
  for (int i = 0; i < x.size(); ++i) {
    auto xi   = x[i];
    auto yi   = y[i];
    auto diff = (xi > yi) ? (xi - yi) : (yi - xi);
    acc += diff * diff * diff;  // |x-y|³
  }
})

int main()
{
  std::cout << "=== cuVS UDF Distance Metrics ===\n\n";

  std::cout << "Defined metrics:\n";
  std::cout << "  1. chebyshev_distance - L∞ norm (max absolute diff)\n";
  std::cout << "  2. weighted_l1        - Weighted L1 distance\n";
  std::cout << "  3. squared_l2         - Standard squared Euclidean\n";
  std::cout << "  4. minkowski_p3       - Minkowski with p=3\n\n";

  std::cout << "Usage:\n";
  std::cout << "  params.udf.metric = chebyshev_distance_udf();\n";
  std::cout << "  params.udf.metric = weighted_l1_udf();\n";
  std::cout << "  params.udf.metric = squared_l2_udf();\n";
  std::cout << "  params.udf.metric = minkowski_p3_udf();\n\n";

  std::cout << "Two approaches for custom metrics:\n";
  std::cout << "  1. Use helpers: acc += cuvs::udf::squared_diff(x, y);\n";
  std::cout << "  2. Use element access: for (int i = 0; i < x.size(); ++i) {...}\n\n";

  std::cout << "Use cases for Chebyshev distance:\n";
  std::cout << "  - Image similarity (max pixel deviation)\n";
  std::cout << "  - Quality control (worst-case tolerance)\n";
  std::cout << "  - Game AI (king's movement on chessboard)\n";

  return 0;
}
