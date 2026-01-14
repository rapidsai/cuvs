/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename OutT, typename LutT>
struct selected {
  dim3 grid_dim;
  dim3 block_dim;
  size_t smem_size;
  size_t device_lut_size;

  selected() = default;

  selected(const void* kernel_ptr, dim3 grid, dim3 block, size_t smem, size_t device_lut)
    : grid_dim(grid),
      block_dim(block),
      smem_size(smem),
      device_lut_size(device_lut),
      kernel(kernel_ptr)
  {
  }

  // This function is ONLY defined in `ivf_pq_compute_similarity_impl.cuh` and compiled in related
  // TUs. Do not re-define it in any other TU as accessing the kernel pointer elsewhere is
  // prohibited per CUDA whole compilation rules. See
  // https://developer.nvidia.com/blog/cuda-c-compiler-updates-impacting-elf-visibility-and-linkage/
  template <typename O, typename L>
  friend auto get_kernel(selected<O, L> s) -> const void*;

 private:
  const void* kernel;  // Type-erased kernel pointer (compatible with any __global__ function)
};

}  // namespace cuvs::neighbors::ivf_pq::detail
