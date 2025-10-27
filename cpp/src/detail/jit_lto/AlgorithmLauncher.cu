/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/AlgorithmLauncher.h>

#include <raft/util/cuda_rt_essentials.hpp>

AlgorithmLauncher::AlgorithmLauncher(cudaKernel_t k) : kernel{k} {}

void AlgorithmLauncher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config;
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.attrs            = attribute;
  config.numAttrs         = 1;
  config.dynamicSmemBytes = shared_mem;

  RAFT_CUDA_TRY(cudaLaunchKernelExC(&config, kernel, kernel_args));
}

std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>>& get_cached_launchers()
{
  static std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>> launchers;
  return launchers;
}
