/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// clang-format off
#include "cuda_stub.hpp"  // must go first
// clang-format on

#include "ann_types.hpp"

#include <dlfcn.h>

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace cuvs::bench {

struct lib_handle {
  void* handle{nullptr};
  explicit lib_handle(const std::string& name)
  {
    handle = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      auto error_msg = "Failed to load " + name;
      auto err       = dlerror();
      if (err != nullptr && err[0] != '\0') { error_msg += ": " + std::string(err); }
      throw std::runtime_error(error_msg);
    }
  }
  ~lib_handle() noexcept
  {
    if (handle != nullptr) { dlclose(handle); }
  }
};

auto load_lib(const std::string& algo) -> void*
{
  static std::unordered_map<std::string, lib_handle> libs{};
  auto found = libs.find(algo);

  if (found != libs.end()) { return found->second.handle; }
  auto lib_name = "lib" + algo + "_ann_bench.so";
  return libs.emplace(algo, lib_name).first->second.handle;
}

/*
  TODO(achirkin): remove this compatibility layer.
  When reading old raft algo configs, we may encounter raft_xxx algorithms;
  they all are renamed to cuvs_xxx algorithm.
  This compatibility layer helps using old configs with the new benchmark executable.
 */
auto load_lib_raft_compat(const std::string& algo) -> void*
{
  try {
    return load_lib(algo);
  } catch (std::runtime_error& e) {
    if (algo.rfind("raft", 0) == 0) { return load_lib("cuvs" + algo.substr(4)); }
    throw e;
  }
}

auto get_fun_name(void* addr) -> std::string
{
  Dl_info dl_info;
  if (dladdr(addr, &dl_info) != 0) {
    if (dl_info.dli_sname != nullptr && dl_info.dli_sname[0] != '\0') {
      return std::string{dl_info.dli_sname};
    }
  }
  throw std::logic_error("Failed to find out name of the looked up function");
}

template <typename T>
auto create_algo(const std::string& algo,
                 const std::string& distance,
                 int dim,
                 const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
  static auto fname = get_fun_name(reinterpret_cast<void*>(&create_algo<T>));
  auto handle       = load_lib_raft_compat(algo);
  auto fun_addr     = dlsym(handle, fname.c_str());
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_algo function (" + algo + ")");
  }
  auto fun = reinterpret_cast<decltype(&create_algo<T>)>(fun_addr);
  return fun(algo, distance, dim, conf);
}

template <typename T>
std::unique_ptr<typename cuvs::bench::algo<T>::search_param> create_search_param(
  const std::string& algo, const nlohmann::json& conf)
{
  static auto fname = get_fun_name(reinterpret_cast<void*>(&create_search_param<T>));
  auto handle       = load_lib_raft_compat(algo);
  auto fun_addr     = dlsym(handle, fname.c_str());
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_search_param function (" + algo + ")");
  }
  auto fun = reinterpret_cast<decltype(&create_search_param<T>)>(fun_addr);
  return fun(algo, conf);
}

};  // namespace cuvs::bench

REGISTER_ALGO_INSTANCE(float);
REGISTER_ALGO_INSTANCE(std::int8_t);
REGISTER_ALGO_INSTANCE(std::uint8_t);

#include "benchmark.hpp"

auto main(int argc, char** argv) -> int { return cuvs::bench::run_main(argc, argv); }
