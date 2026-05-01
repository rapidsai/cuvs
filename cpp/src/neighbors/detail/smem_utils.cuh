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
#include <utility>

namespace cuvs::neighbors::detail {

/** Host-side pointer to @p KernelT (or nullptr for pure JIT) paired with the CUDA module kernel
 *  handle; used to track which logical kernel last had smem attributes applied. */
template <typename KernelT>
using kernel_cuda_pair_t = std::pair<std::add_pointer_t<KernelT>, cudaKernel_t>;

/**
 * @brief (Thread-)Safely invoke a kernel with a maximum dynamic shared memory size.
 *
 * Maintains a monotonically growing high-water mark for
 * `cudaFuncAttributeMaxDynamicSharedMemorySize`. When the kernel identity changes, the new kernel
 * is brought up to the current high-water mark; when @p smem_size exceeds the high-water mark, it
 * is grown for the current kernel. This guarantees every kernel's attribute is always >= @p
 * smem_size at the time of launch.
 *
 * This is required because the sequence `cudaFuncSetAttribute` + kernel launch is not executed
 * atomically. Used this way, `cudaFuncAttributeMaxDynamicSharedMemorySize` can only grow and the
 * kernel remains safe to launch.
 *
 * NB: cudaFuncSetAttribute is per kernel handle value, not per C++ type. Multiple template
 * instantiations may share the same @p KernelT (e.g. the same function signature), so we track the
 * kernel identity alongside the smem high-water mark using @p cuda_kernel (and optional host
 * function pointer in the pair's first element).
 *
 * @tparam KernelT Kernel function type from kernel_def.hpp (keys static state per signature).
 * @tparam KernelLauncherT Type of the launch callable (e.g. lambda calling launcher->dispatch).
 * @param smem_size Dynamic shared memory required for this launch.
 * @param launch Invoked after attributes are set; takes no arguments.
 * @param cuda_kernel Handle passed to cudaFuncSetAttribute (e.g. launcher->get_kernel()). Pure JIT
 *                    callers pair this with `nullptr` as the host pointer.
 */
template <typename KernelT, typename KernelLauncherT>
void safely_launch_kernel_with_smem_size(std::uint32_t smem_size,
                                         KernelLauncherT const& launch,
                                         cudaKernel_t cuda_kernel)
{
  // last_smem_size is a monotonically growing high-water mark. last_kernel is a (host fn ptr,
  // cudaKernel_t) pair tracking which kernel identity was last configured.
  static std::atomic<std::uint32_t> last_smem_size{0};
  static std::atomic<kernel_cuda_pair_t<KernelT>> last_kernel{
    kernel_cuda_pair_t<KernelT>{nullptr, cudaKernel_t{}}};
  static std::mutex mutex;

  kernel_cuda_pair_t<KernelT> const current{nullptr, cuda_kernel};

  // Fast path: skip the lock when the kernel identity matches and smem is within bounds.
  // Load order matters: last_smem_size (acquire) before last_kernel (relaxed). Inside the lock we
  // store in the opposite order: last_kernel (relaxed) then last_smem_size (release). This way an
  // acquire load of last_smem_size that sees a post-cudaFuncSetAttribute value is guaranteed to
  // also see the corresponding last_kernel pair.
  std::uint32_t observed_smem                 = last_smem_size.load(std::memory_order_acquire);
  kernel_cuda_pair_t<KernelT> observed_kernel = last_kernel.load(std::memory_order_relaxed);
  if (smem_size > observed_smem || current != observed_kernel) {
    std::lock_guard<std::mutex> guard(mutex);
    // Re-check under the lock: the outside decision can be stale.
    std::uint32_t cur_smem_size = last_smem_size.load(std::memory_order_relaxed);
    observed_kernel             = last_kernel.load(std::memory_order_relaxed);
    bool need_update            = (current != observed_kernel);
    if (smem_size > cur_smem_size) {
      cur_smem_size = smem_size;
      need_update   = true;
    }
    if (need_update) {
      auto launch_status = cudaFuncSetAttribute(
        cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cur_smem_size);
      RAFT_EXPECTS(launch_status == cudaSuccess,
                   "Failed to set max dynamic shared memory size to %u bytes",
                   cur_smem_size);
      // Store order matters: last_kernel before last_smem_size (release) so the fast-path acquire
      // load of last_smem_size also publishes last_kernel.
      last_kernel.store(current, std::memory_order_relaxed);
      last_smem_size.store(cur_smem_size, std::memory_order_release);
    }
  }
  return launch();
}

}  // namespace cuvs::neighbors::detail
