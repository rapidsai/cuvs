/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>

namespace cuvs::neighbors::detail {

template <typename KernelT, typename KernelLauncherT>
void safely_launch_kernel_with_smem_size_impl(KernelT const& kernel,
                                              uint32_t smem_size,
                                              KernelLauncherT const& launch,
                                              std::mutex& mutex,
                                              std::atomic<uint32_t>& current_smem_size)
{
  auto last_smem_size = current_smem_size.load(std::memory_order_relaxed);
  if (smem_size > last_smem_size) {
    // We still need a mutex for the critical section: actualize last_smem_size and set the
    // attribute.
    auto guard = std::lock_guard<std::mutex>{mutex};
    if (!current_smem_size.compare_exchange_strong(
          last_smem_size, smem_size, std::memory_order_relaxed, std::memory_order_relaxed)) {
      // The value has been updated by another thread between the load and the mutex acquisition.
      if (smem_size > last_smem_size) {
        current_smem_size.store(smem_size, std::memory_order_relaxed);
      }
    }
    // Only update if the last seen value is smaller than the new one.
    if (smem_size > last_smem_size) {
      auto launch_status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      RAFT_EXPECTS(launch_status == cudaSuccess,
                   "Failed to set max dynamic shared memory size to %u bytes",
                   smem_size);
    }
  }
  // We don't need to guard the kernel launch because the smem_size can only grow.
  return launch(kernel);
}

/**
 * @brief (Thread-)Safely invoke a kernel with a maximum dynamic shared memory size.
 * This is required because the sequence `cudaFuncSetAttribute` + kernel launch is not executed
 * atomically.
 *
 * Used this way, the cudaFuncAttributeMaxDynamicSharedMemorySize can only grow and thus
 * guarantees that the kernel is safe to launch.
 *
 * @tparam KernelT The type of the kernel.
 * @tparam InvocationT The type of the invocation function.
 * @param kernel The kernel function address (for whom the smem-size is specified).
 * @param smem_size The size of the dynamic shared memory to be set.
 * @param launch The kernel launch function/lambda.
 */
// Specialization for cudaKernel_t (JIT LTO kernels) - track by kernel pointer
template <typename KernelLauncherT>
void safely_launch_kernel_with_smem_size(cudaKernel_t kernel,
                                         uint32_t smem_size,
                                         KernelLauncherT const& launch)
{
  // For JIT kernels, track by kernel pointer since all cudaKernel_t have the same type
  static std::unordered_map<cudaKernel_t, std::pair<std::mutex, std::atomic<uint32_t>>>
    jit_smem_sizes;
  std::mutex map_mutex;

  std::pair<std::mutex, std::atomic<uint32_t>>* current_smem_size;
  {
    std::lock_guard<std::mutex> map_lock{map_mutex};
    current_smem_size = &jit_smem_sizes[kernel];
  }
  safely_launch_kernel_with_smem_size_impl<cudaKernel_t, KernelLauncherT>(
    kernel, smem_size, launch, current_smem_size->first, current_smem_size->second);
}

// General template for regular function pointers
template <typename KernelT, typename KernelLauncherT>
void safely_launch_kernel_with_smem_size(KernelT const& kernel,
                                         uint32_t smem_size,
                                         KernelLauncherT const& launch)
{
  // the last smem size is parameterized by the kernel thanks to the template parameter.
  static std::atomic<uint32_t> current_smem_size{0};
  static std::mutex mutex;

  safely_launch_kernel_with_smem_size_impl<KernelT, KernelLauncherT>(
    kernel, smem_size, launch, mutex, current_smem_size);
}

}  // namespace cuvs::neighbors::detail
