/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file compiler.cpp
 * @brief Implementation of UDF JIT compiler using NVRTC
 *
 * This file shows how cuVS implements JIT compilation of user-defined metrics.
 * Key responsibility: append the compute_dist wrapper and explicit instantiation
 * to the user's struct definition.
 */

#include <cuvs/udf/compiler.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <functional>
#include <sstream>
#include <stdexcept>

namespace cuvs::udf::detail {

// ============================================================
// Cache Implementation
// ============================================================

size_t cache_key_hash::operator()(const cache_key& k) const
{
  size_t h = 0;
  auto hash_combine = [&h](const auto& v) {
    h ^= std::hash<std::decay_t<decltype(v)>>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
  };

  hash_combine(k.source_hash);
  hash_combine(k.struct_name);
  hash_combine(k.veclen);
  hash_combine(k.data_type);
  hash_combine(k.acc_type);
  hash_combine(k.compute_capability);

  return h;
}

udf_cache& udf_cache::instance()
{
  static udf_cache cache;
  return cache;
}

std::shared_ptr<compiled_fragment> udf_cache::get(const cache_key& key)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = cache_.find(key);
  if (it != cache_.end()) { return it->second; }
  return nullptr;
}

void udf_cache::put(const cache_key& key, std::shared_ptr<compiled_fragment> fragment)
{
  std::lock_guard<std::mutex> lock(mutex_);
  cache_[key] = std::move(fragment);
}

void udf_cache::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

// ============================================================
// Source Building - THE KEY FUNCTION
// ============================================================

std::string build_full_source(const metric_source& udf,
                              int veclen,
                              const std::string& data_type,
                              const std::string& acc_type)
{
  std::stringstream ss;

  // 1. Standard includes
  ss << "#include <cuda_fp16.h>\n";
  ss << "#include <cstdint>\n";
  ss << "#include <type_traits>\n\n";

  // 2. Include the point wrapper and metric interface
  ss << "#include <cuvs/udf/point.cuh>\n";
  ss << "#include <cuvs/udf/metric_interface.cuh>\n\n";

  // 3. Open namespace
  ss << "namespace cuvs::neighbors::ivf_flat::detail {\n\n";

  // 4. User's struct definition (from metric_source.source)
  //    This is ONLY the struct - no wrapper, no instantiation
  ss << "// User-defined metric struct\n";
  ss << udf.source << "\n\n";

  // 5. cuVS adds the compute_dist wrapper function
  //    This calls the user's struct with point-wrapped arguments
  ss << "// cuVS-generated wrapper function\n";
  ss << "template <int Veclen, typename T, typename AccT>\n";
  ss << "__device__ void compute_dist(AccT& acc, AccT x_raw, AccT y_raw) {\n";
  ss << "    // Wrap raw values in point<T, AccT, Veclen>\n";
  ss << "    using point_t = cuvs::udf::point<T, AccT, Veclen>;\n";
  ss << "    point_t x{x_raw};\n";
  ss << "    point_t y{y_raw};\n";
  ss << "    " << udf.struct_name << "<T, AccT, Veclen>{}(acc, x, y);\n";
  ss << "}\n\n";

  // 6. cuVS adds the explicit instantiation
  //    Based on index.veclen() and index.data_type()
  ss << "// cuVS-generated explicit instantiation\n";
  ss << "template __device__ void compute_dist<" << veclen << ", " << data_type << ", " << acc_type
     << ">(" << acc_type << "&, " << acc_type << ", " << acc_type << ");\n\n";

  // 7. Close namespace
  ss << "}  // namespace cuvs::neighbors::ivf_flat::detail\n";

  return ss.str();
}

// ============================================================
// Hash helper
// ============================================================

static std::string compute_source_hash(const std::string& source)
{
  std::hash<std::string> hasher;
  return std::to_string(hasher(source));
}

// ============================================================
// NVRTC Error Checking
// ============================================================

static void check_nvrtc(nvrtcResult result, const char* msg)
{
  if (result != NVRTC_SUCCESS) {
    std::stringstream ss;
    ss << msg << ": " << nvrtcGetErrorString(result);
    throw compilation_error(ss.str());
  }
}

// ============================================================
// Main Compilation Function
// ============================================================

std::shared_ptr<compiled_fragment> compile_metric(const metric_source& udf,
                                                  int veclen,
                                                  const std::string& data_type,
                                                  const std::string& acc_type)
{
  // 1. Get device compute capability
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);
  int cc = props.major * 10 + props.minor;

  // 2. Check cache first
  cache_key key{.source_hash        = compute_source_hash(udf.source),
                .struct_name        = udf.struct_name,
                .veclen             = veclen,
                .data_type          = data_type,
                .acc_type           = acc_type,
                .compute_capability = cc};

  auto& cache = udf_cache::instance();
  if (auto cached = cache.get(key)) { return cached; }

  // 3. Build full source (user struct + wrapper + instantiation)
  std::string full_source = build_full_source(udf, veclen, data_type, acc_type);

  // 4. Prepare headers for NVRTC (include point.cuh and metric_interface.cuh)
  std::vector<const char*> header_names;
  std::vector<const char*> header_contents;

  for (const auto& [name, content] : udf.headers) {
    header_names.push_back(name.c_str());
    header_contents.push_back(content.c_str());
  }

  // 5. Create NVRTC program
  nvrtcProgram prog;
  check_nvrtc(nvrtcCreateProgram(&prog,
                                 full_source.c_str(),
                                 "udf_metric.cu",
                                 static_cast<int>(header_names.size()),
                                 header_contents.data(),
                                 header_names.data()),
              "Failed to create NVRTC program");

  // 6. Compile options for LTO
  std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(cc);

  const char* opts[] = {
    arch_opt.c_str(),
    "-dlto",  // Generate LTO-IR
    "--relocatable-device-code=true",
    "-std=c++17",
    "-default-device",
  };

  nvrtcResult compile_result = nvrtcCompileProgram(prog, 5, opts);

  // 7. Get compilation log
  size_t log_size;
  nvrtcGetProgramLogSize(prog, &log_size);

  std::string log;
  if (log_size > 1) {
    log.resize(log_size);
    nvrtcGetProgramLog(prog, log.data());
  }

  if (compile_result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    throw compilation_error("UDF compilation failed:\n" + log);
  }

  // 8. Get LTO-IR
  size_t lto_size;
  check_nvrtc(nvrtcGetLTOIRSize(prog, &lto_size), "Failed to get LTO-IR size");

  auto fragment = std::make_shared<compiled_fragment>();
  fragment->lto_ir.resize(lto_size);
  check_nvrtc(nvrtcGetLTOIR(prog, fragment->lto_ir.data()), "Failed to get LTO-IR");

  nvrtcDestroyProgram(&prog);

  // 9. Cache and return
  cache.put(key, fragment);

  return fragment;
}

}  // namespace cuvs::udf::detail
