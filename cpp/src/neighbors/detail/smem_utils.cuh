/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/error.hpp>

#include <cstdint>
#include <mutex>

namespace cuvs::neighbors::detail {

/**
 * @brief (Thread-)Safely invoke a kernel with a maximum dynamic shared memory size.
 *
 * Maintains a monotonically growing high-water mark for `cudaFuncAttributeMaxDynamicSharedMemorySize`.
 * When the kernel function pointer changes, the new kernel is brought up to the current high-water
 * mark; when smem_size exceeds the high-water mark, it is grown for the current kernel.
 * This guarantees every kernel's attribute is always >= smem_size at the time of launch.
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
  // current_smem_size is a monotonically growing high-water mark across all kernel pointers.
  // current_kernel tracks which kernel pointer was last used.
  static uint32_t current_smem_size{0};
  static KernelT current_kernel{KernelT{}};
  static std::mutex mutex;

  {
    std::lock_guard<std::mutex> guard(mutex);

    auto last_kernel    = current_kernel;
    auto last_smem_size = current_smem_size;

    // When the kernel function pointer changes, bring the new kernel up to the global high-water
    // mark. This is necessary because cudaFuncSetAttribute applies to a specific function pointer,
    // not to the pointer type — different template instantiations may share the same KernelT.
    if (kernel != last_kernel) {
      current_kernel     = kernel;
      auto launch_status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, last_smem_size);
      RAFT_EXPECTS(launch_status == cudaSuccess,
                   "Failed to set max dynamic shared memory size to %u bytes",
                   last_smem_size);
    }
    // When smem_size exceeds the high-water mark, grow it for the current kernel.
    // If the kernel also changed above, this handles the case where smem_size > last_smem_size.
    if (smem_size > last_smem_size) {
      current_smem_size  = smem_size;
      auto launch_status =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
      RAFT_EXPECTS(launch_status == cudaSuccess,
                   "Failed to set max dynamic shared memory size to %u bytes",
                   smem_size);
    }
  }
  // The kernel launch is outside the lock: any concurrent cudaFuncSetAttribute can only increase
  // the limit, so the launch is always safe.
  return launch(kernel);
}

}  // namespace cuvs::neighbors::detail
