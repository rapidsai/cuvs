/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <raft/core/error.hpp>

namespace cuvs::neighbors::detail {

/**
 * @brief (Thread-)Safely invoke a kernel with a maximum dynamic shared memory size.
 *
 * Maintains a monotonically growing high-water mark for
 * `cudaFuncAttributeMaxDynamicSharedMemorySize`. When the kernel function pointer changes, the new
 * kernel is brought up to the current high-water mark; when smem_size exceeds the high-water mark,
 * it is grown for the current kernel. This guarantees every kernel's attribute is always >=
 * smem_size at the time of launch.
 *
 * NB: cudaFuncSetAttribute is per kernel function pointer value, not per type. Multiple kernel
 * template instantiations may share the same KernelT type (e.g. function pointers with the same
 * signature), so we track the kernel identity alongside the smem high-water mark.
 *
 * @tparam KernelT The type of the kernel.
 * @tparam KernelLauncherT The type of the launch function/lambda.
 * @param kernel The kernel function address (for whom the smem-size is specified).
 * @param smem_size The size of the dynamic shared memory to be set.
 * @param launch The kernel launch function/lambda.
 */
template <typename KernelT, typename KernelLauncherT>
void safely_launch_kernel_with_smem_size_impl(KernelT const& kernel,
                                              uint32_t smem_size,
                                              KernelLauncherT const& launch,
                                              std::mutex& mutex,
                                              std::atomic<uint32_t>& current_smem_size)
{
  // last_smem_size is a monotonically growing high-water mark across all kernel pointers.
  // last_kernel tracks which kernel pointer was last used.
  static std::atomic<uint32_t> last_smem_size{0};
  static std::atomic<KernelT> last_kernel{KernelT{}};
  static std::mutex mutex;
  // Fast path: skip the lock when the kernel matches and the smem size is within bounds.
  // Load order matters: last_smem_size (acquire) before last_kernel (relaxed). Inside the lock
  // we store in the opposite order: last_kernel (relaxed) then last_smem_size (release).
  // This way an acquire load of last_smem_size that sees a post-cudaFuncSetAttribute value is
  // guaranteed to also see the corresponding last_kernel.
  if (smem_size > last_smem_size.load(std::memory_order_acquire) ||
      kernel != last_kernel.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> guard(mutex);
    // Re-check under the lock: the outside decision can be stale.
    uint32_t cur_smem_size = last_smem_size.load(std::memory_order_relaxed);
    bool need_update       = (kernel != last_kernel.load(std::memory_order_relaxed));
    if (smem_size > cur_smem_size) {
      cur_smem_size = smem_size;
      need_update   = true;
    }
    if (need_update) {
      auto launch_status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cur_smem_size);
      RAFT_EXPECTS(launch_status == cudaSuccess,
                   "Failed to set max dynamic shared memory size to %u bytes",
                   cur_smem_size);
      // Store order matters: last_kernel before last_smem_size (release) so the fast-path
      // acquire load of last_smem_size also publishes last_kernel.
      last_kernel.store(kernel, std::memory_order_relaxed);
      last_smem_size.store(cur_smem_size, std::memory_order_release);
    }
  }
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
