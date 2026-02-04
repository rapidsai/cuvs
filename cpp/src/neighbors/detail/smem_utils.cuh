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
 * @brief Optionally set the larger max dynamic shared memory size for the kernel.
 * This is required because `cudaFuncSetAttribute` is not thread-safe.
 * In the event of concurrent calls, we'd like to accommodate the largest requested size.
 * @tparam KernelT The type of the kernel.
 * @param smem_size The size of the dynamic shared memory to be set.
 * @param kernel The kernel to be set.
 */
template <typename KernelT>
void optionally_set_larger_max_smem_size(uint32_t smem_size, KernelT& kernel)
{
  static auto mutex                 = std::mutex{};
  static auto running_max_smem_size = uint32_t{0};
  if (smem_size > running_max_smem_size) {
    auto guard = std::lock_guard<std::mutex>{mutex};
    if (smem_size > running_max_smem_size) {
      running_max_smem_size = smem_size;
      RAFT_CUDA_TRY(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, running_max_smem_size));
    }
  }
}

}  // namespace cuvs::neighbors::detail
