/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cuvs::udf {

/**
 * @brief Source definition for a custom metric.
 *
 * Contains the struct source code and metadata needed for JIT compilation.
 * Typically created via the CUVS_METRIC macro (see metric_macro.hpp).
 */
struct metric_source {
  /**
   * @brief CUDA source code containing the metric struct.
   *
   * Should define a template struct:
   *   template <typename T, typename AccT, int Veclen>
   *   struct your_name {
   *       __device__ void operator()(AccT& acc, AccT x, AccT y) { ... }
   *   };
   *
   * Note: Do NOT include the compute_dist wrapper or explicit instantiation.
   *       cuVS appends those automatically based on index properties.
   */
  std::string source;

  /**
   * @brief Name of the metric struct (without template parameters).
   *
   * cuVS uses this to generate:
   *   your_name<T, AccT, Veclen>{}(acc, x, y);
   */
  std::string struct_name;

  /**
   * @brief Optional headers the metric depends on.
   *
   * Map of header name -> header content.
   * Passed to NVRTC's virtual filesystem.
   */
  std::unordered_map<std::string, std::string> headers;
};

/**
 * @brief UDF configuration for search parameters.
 */
struct udf_config {
  std::optional<metric_source> metric;

  // Future extensions:
  // std::optional<filter_source> sample_filter;
  // std::optional<postprocess_source> post_process;
};

/**
 * @brief Exception thrown when UDF JIT compilation fails.
 */
class compilation_error : public std::runtime_error {
 public:
  explicit compilation_error(const std::string& msg) : std::runtime_error(msg) {}
};

}  // namespace cuvs::udf
