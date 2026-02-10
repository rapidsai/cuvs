/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cuda_rt_essentials.hpp>

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
 * @param invoke_kernel The kernel invocation function/lambda.
 */
template <typename KernelT, typename InvocationT>
void safely_invoke_kernel_with_smem_size(KernelT& kernel,
                                         uint32_t smem_size,
                                         InvocationT const& invoke_kernel)
{
  static auto mutex = std::mutex{};
  auto guard        = std::lock_guard<std::mutex>{mutex};
  RAFT_CUDA_TRY(
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  invoke_kernel();
}

}  // namespace cuvs::neighbors::detail
