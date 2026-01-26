/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "metric_source.hpp"

namespace cuvs::udf::detail {

/**
 * @brief Compiled UDF fragment (LTO-IR).
 */
struct compiled_fragment {
  std::vector<char> lto_ir;
  size_t size() const { return lto_ir.size(); }
  const char* data() const { return lto_ir.data(); }
};

/**
 * @brief Cache key for compiled UDFs.
 */
struct cache_key {
  std::string source_hash;
  std::string struct_name;
  int veclen;
  std::string data_type;
  std::string acc_type;
  int compute_capability;

  bool operator==(const cache_key& other) const
  {
    return source_hash == other.source_hash && struct_name == other.struct_name &&
           veclen == other.veclen && data_type == other.data_type && acc_type == other.acc_type &&
           compute_capability == other.compute_capability;
  }
};

struct cache_key_hash {
  size_t operator()(const cache_key& k) const;
};

/**
 * @brief Thread-safe cache for compiled UDF fragments.
 */
class udf_cache {
 public:
  static udf_cache& instance();

  std::shared_ptr<compiled_fragment> get(const cache_key& key);
  void put(const cache_key& key, std::shared_ptr<compiled_fragment> fragment);
  void clear();

 private:
  udf_cache() = default;
  std::unordered_map<cache_key, std::shared_ptr<compiled_fragment>, cache_key_hash> cache_;
  std::mutex mutex_;
};

/**
 * @brief Build the full source code for JIT compilation.
 *
 * Takes the user's struct source and appends:
 * 1. Standard includes
 * 2. The compute_dist wrapper function
 * 3. The explicit instantiation for the given Veclen/T/AccT
 *
 * @param udf User's metric source (struct only)
 * @param veclen Vector length from index
 * @param data_type Data type string (e.g., "float")
 * @param acc_type Accumulator type string (e.g., "float")
 * @return Complete source ready for NVRTC
 */
std::string build_full_source(const metric_source& udf,
                              int veclen,
                              const std::string& data_type,
                              const std::string& acc_type);

/**
 * @brief Compile a UDF metric source to LTO-IR.
 *
 * @param udf The user's metric source
 * @param veclen Vector length from index
 * @param data_type Data type string
 * @param acc_type Accumulator type string
 * @return Compiled fragment (LTO-IR)
 * @throws compilation_error if NVRTC compilation fails
 */
std::shared_ptr<compiled_fragment> compile_metric(const metric_source& udf,
                                                  int veclen,
                                                  const std::string& data_type,
                                                  const std::string& acc_type);

}  // namespace cuvs::udf::detail
