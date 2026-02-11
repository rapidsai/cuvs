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
 * @brief (Thread-)Safely invoke a kernel with a dynamic shared memory size.
 * This is required because `cudaFuncSetAttribute` is not thread-safe.
 * In the event of concurrent calls, each kernel will be registered to run with
 * its specified smem-size.
 * @tparam KernelT The type of the kernel.
 * @tparam InvocationT The type of the invocation function.
 * @param kernel The kernel function address (for whom the smem-size is specified).
 * @param smem_size The size of the dynamic shared memory to be set.
 * @param launch The kernel launch function/lambda.
 */
template <typename KernelT, typename KernelLauncherT>
void safely_launch_kernel_with_smem_size(KernelT const& kernel,
                                         uint32_t smem_size,
                                         KernelLauncherT const& launch)
{
  static auto mutex = std::mutex{};
  auto guard        = std::lock_guard<std::mutex>{mutex};
  auto launch_status =
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  RAFT_EXPECTS(launch_status == cudaSuccess,
               "Failed to set max dynamic shared memory size to %zu bytes",
               smem_size);
  launch(kernel);
}

}  // namespace cuvs::neighbors::detail
