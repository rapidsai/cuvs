/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/failure_callback_resource_adaptor.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
#include <optional>

/** Report a more verbose error with a backtrace when OOM occurs on RMM side. */
inline auto rmm_oom_callback(std::size_t bytes, void*) -> bool
{
  auto cuda_status = cudaGetLastError();
  size_t free      = 0;
  size_t total     = 0;
  RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
  RAFT_FAIL(
    "[cuVS Performance] Failed to allocate %zu bytes using RMM memory resource. "
    "NB: latest cuda status = %s, free memory = %zu, total memory = %zu.",
    bytes,
    cudaGetErrorName(cuda_status),
    free,
    total);
}

/** Helper class to setup a pool memory resource for a single device. */
class global_mem_resource {
 public:
  using pool_mr_type  = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  using mr_type       = rmm::mr::failure_callback_resource_adaptor<pool_mr_type>;
  using large_mr_type = rmm::mr::managed_memory_resource;

  constexpr static size_t kInitialSize           = 1024ull * 1024ull * 1024ull;
  constexpr static double kMaxMemoryUsage        = 0.8;
  constexpr static double kMaxInitialMemoryUsage = 0.5;

  global_mem_resource()
  try
    : orig_resource_{rmm::mr::get_current_device_resource()},
      pool_resource_(orig_resource_, compute_initial_size(), compute_max_size()),
      resource_(&pool_resource_, rmm_oom_callback, nullptr) {
    rmm::mr::set_current_device_resource(&resource_);
  } catch (const std::exception& e) {
    auto cuda_status = cudaGetLastError();
    size_t free      = 0;
    size_t total     = 0;
    RAFT_CUDA_TRY_NO_THROW(cudaMemGetInfo(&free, &total));
    RAFT_FAIL(
      "Failed to initialize shared raft resources (NB: latest cuda status = %s, free memory = %zu, "
      "total memory = %zu): %s",
      cudaGetErrorName(cuda_status),
      free,
      total,
      e.what());
  }

  global_mem_resource(global_mem_resource&&)                               = delete;
  auto operator=(global_mem_resource&&) -> global_mem_resource&            = delete;
  global_mem_resource(const global_mem_resource& res)                      = delete;
  auto operator=(const global_mem_resource& other) -> global_mem_resource& = delete;

  ~global_mem_resource() noexcept { rmm::mr::set_current_device_resource(orig_resource_); }

 private:
  rmm::mr::device_memory_resource* orig_resource_;
  pool_mr_type pool_resource_;
  mr_type resource_;

  static auto compute_initial_size() -> size_t
  {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { free_bytes = kInitialSize; }
    auto limit = static_cast<size_t>(free_bytes * kMaxInitialMemoryUsage);
    return raft::round_up_safe<size_t>(std::min<size_t>(kInitialSize, limit), 256ull);
  }

  static auto compute_max_size() -> size_t
  {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { total_bytes = kInitialSize; }
    return raft::round_up_safe<size_t>(total_bytes * kMaxMemoryUsage, 256ull);
  }
};

/** Remember and restore the current device. */
struct keep_current_device_raii {
  int initial_device_id = 0;
  keep_current_device_raii() { cudaGetDevice(&initial_device_id); }
  ~keep_current_device_raii() { cudaSetDevice(initial_device_id); }
};

/** Handles for each RMM memory resource. */
static std::array<std::optional<global_mem_resource>, 8> global_mem_resource_;

// Initialize at the moment libcuvs.so is loaded.
__attribute__((constructor)) void cuvs_performance_init()
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) { return; }

  // Start CUDA profiler (ignore errors if not supported in this context)
  (void)cudaProfilerStart();

  keep_current_device_raii keep_current_device;

  // == DISABLED FOR NOW TO TEST ACTUAL PERFORMANCE ==
  // Configure each device with a pool memory resource
  // for (int device_id = 0; device_id < device_count; ++device_id) {
  //   if (cudaSetDevice(device_id) != cudaSuccess) { continue; }
  //   global_mem_resource_[device_id].emplace();
  // }
}

// Cleanup before unloading libcuvs.so.
__attribute__((destructor)) void cuvs_performance_cleanup()
{
  (void)cudaProfilerStop();

  keep_current_device_raii keep_current_device;
  for (auto& global_mem_resource : global_mem_resource_) {
    global_mem_resource.reset();
  }
}
