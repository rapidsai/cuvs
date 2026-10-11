/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime_api.h>

#include <mutex>
#include <unordered_map>

namespace cuvs::util::detail {

/**
 * Whether `device` reads pageable (unregistered) host memory through the host page tables (ATS,
 * e.g. Grace Hopper / Grace Blackwell). The result is cached per device.
 */
inline auto pageable_memory_uses_host_page_tables(int device) -> bool
{
  static std::mutex mutex;
  static std::unordered_map<int, bool> cache;
  std::lock_guard<std::mutex> lock(mutex);
  if (auto it = cache.find(device); it != cache.end()) { return it->second; }
  int uses_host_page_tables = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(
    &uses_host_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, device));
  cache.emplace(device, uses_host_page_tables != 0);
  return uses_host_page_tables != 0;
}

/**
 * Whether memory described by `attr` may be dereferenced in device code on the current device.
 *
 * Device and managed memory are device-accessible, and registered (pinned) host memory is when it
 * is mapped (`devicePointer` is set). Pageable (unregistered) host memory is device-accessible only
 * when the device reads it through the host page tables (ATS). On systems with HMM,
 * `cudaPointerGetAttributes` can also report a non-null `devicePointer` for pageable memory (for
 * example after the first GPU access to pageable memory in the process), but device accesses to it
 * are then serviced through GPU page faults, so it is treated as host-only there.
 */
inline auto is_device_accessible(const cudaPointerAttributes& attr) -> bool
{
  switch (attr.type) {
    case cudaMemoryTypeDevice:
    case cudaMemoryTypeManaged: return true;
    case cudaMemoryTypeHost: return attr.devicePointer != nullptr;
    default: {
      if (attr.devicePointer == nullptr) { return false; }
      int device = 0;
      RAFT_CUDA_TRY(cudaGetDevice(&device));
      return pageable_memory_uses_host_page_tables(device);
    }
  }
}

}  // namespace cuvs::util::detail
