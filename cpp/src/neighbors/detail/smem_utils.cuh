/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <atomic>
#include <cstdint>
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
void safely_launch_kernel_with_smem_size(KernelT const& kernel,
                                         uint32_t smem_size,
                                         KernelLauncherT const& launch)
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

}  // namespace cuvs::neighbors::detail
