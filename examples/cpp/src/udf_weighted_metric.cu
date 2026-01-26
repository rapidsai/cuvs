/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file udf_weighted_metric.cu
 * @brief Example: Custom metric with helper headers
 *
 * This example shows how to use custom headers with your UDF metric.
 * Headers are passed to NVRTC's virtual filesystem.
 */

#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/udf/metric_source.hpp>
#include <raft/core/device_resources.hpp>

#include <iostream>

int main()
{
  raft::device_resources res;

  // ============================================================
  // Define helper header
  // ============================================================
  //
  // If your metric needs helper functions or constants,
  // you can provide them as headers.

  std::string math_utils_header = R"(
        #pragma once
        
        namespace my_utils {
        
        template <typename T>
        __device__ __forceinline__ T safe_abs(T x) {
            return (x < T{0}) ? -x : x;
        }
        
        template <typename T>
        __device__ __forceinline__ T clamp(T x, T lo, T hi) {
            return (x < lo) ? lo : ((x > hi) ? hi : x);
        }
        
        // Custom weight function - could be learned from data!
        template <typename T>
        __device__ __forceinline__ T importance_weight() {
            return T{2.5};
        }
        
        }  // namespace my_utils
    )";

  // ============================================================
  // Define metric using the helper header and point wrapper
  // ============================================================

  cuvs::udf::metric_source weighted_metric = {
    .source = R"(
            #include "math_utils.cuh"
            #include <cuvs/udf/metric_interface.cuh>
            
            template <typename T, typename AccT, int Veclen>
            struct weighted_euclidean 
                : cuvs::udf::metric_interface<T, AccT, Veclen> 
            {
                using point_type = cuvs::udf::point<T, AccT, Veclen>;
                
                __device__ void operator()(AccT& acc, point_type x, point_type y) override {
                    // Use helper for optimal squared diff
                    auto sq_diff = cuvs::udf::squared_diff(x, y);
                    
                    // Apply custom weight
                    auto weight = my_utils::importance_weight<AccT>();
                    acc += weight * sq_diff;
                }
            };
        )",
    .struct_name = "weighted_euclidean",

    // Provide the header content
    .headers = {{"math_utils.cuh", math_utils_header}}};

  // ============================================================
  // Alternative: Per-dimension weights using element access
  // ============================================================

  cuvs::udf::metric_source per_dim_weighted = {
    .source = R"(
            #include <cuvs/udf/metric_interface.cuh>
            
            template <typename T, typename AccT, int Veclen>
            struct per_dim_weighted_l2
                : cuvs::udf::metric_interface<T, AccT, Veclen> 
            {
                using point_type = cuvs::udf::point<T, AccT, Veclen>;
                
                __device__ void operator()(AccT& acc, point_type x, point_type y) override {
                    // Per-dimension weights using element access
                    for (int i = 0; i < x.size(); ++i) {
                        auto diff = x[i] - y[i];
                        auto weight = AccT{1} + AccT{i} * AccT{0.1};  // Increasing weights
                        acc += weight * diff * diff;
                    }
                }
            };
        )",
    .struct_name = "per_dim_weighted_l2",
    .headers     = {}};

  // ============================================================
  // Search configuration
  // ============================================================

  cuvs::neighbors::ivf_flat::search_params params;
  params.n_probes = 50;
  // params.udf.metric = weighted_metric;

  std::cout << "Weighted Euclidean distance metric example!\n";
  std::cout << "\n";
  std::cout << "This demonstrates:\n";
  std::cout << "  1. Using custom helper headers with UDFs\n";
  std::cout << "  2. Using cuvs::udf::squared_diff(x, y) helper\n";
  std::cout << "  3. Per-dimension weights using x[i], y[i] element access\n";
  std::cout << "\n";
  std::cout << "The point<T, AccT, Veclen> wrapper provides:\n";
  std::cout << "  - squared_diff(x, y) : optimal for all types\n";
  std::cout << "  - x[i], y[i]         : element access for custom logic\n";
  std::cout << "  - x.raw()            : raw storage for power users\n";

  return 0;
}
