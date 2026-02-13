/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

AlgorithmLauncher::AlgorithmLauncher(cudaKernel_t k) : kernel{k} {}

void AlgorithmLauncher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  // Debug: verify kernel is being called
  if (kernel != nullptr) {
    std::cerr << "[JIT] AlgorithmLauncher::call - kernel is not null, launching with grid=("
              << grid.x << "," << grid.y << "," << grid.z << ") block=(" << block.x << ","
              << block.y << "," << block.z << ")" << std::endl;
    std::cerr.flush();
  } else {
    std::cerr << "[JIT] ERROR: AlgorithmLauncher::call - kernel is NULL!" << std::endl;
    std::cerr.flush();
  }

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

  std::cerr << "[JIT] AlgorithmLauncher::call - About to launch kernel" << std::endl;
  std::cerr.flush();

  cudaError_t err = cudaLaunchKernelExC(&config, kernel, kernel_args);
  if (err != cudaSuccess) {
    std::cerr << "[JIT] ERROR: cudaLaunchKernelExC failed with: " << cudaGetErrorString(err) << " ("
              << err << ")" << std::endl;
    std::cerr.flush();
  } else {
    std::cerr << "[JIT] Kernel launch succeeded" << std::endl;
    std::cerr.flush();
  }
  RAFT_CUDA_TRY(err);
}

void AlgorithmLauncher::call_cooperative(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  cudaLaunchAttribute attributes[2];
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  attributes[1].id                                         = cudaLaunchAttributeCooperative;
  attributes[1].val.cooperative                            = 1;

  cudaLaunchConfig_t config;
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.attrs            = attributes;
  config.numAttrs         = 2;
  config.dynamicSmemBytes = shared_mem;

  RAFT_CUDA_TRY(cudaLaunchKernelExC(&config, kernel, kernel_args));
}

std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>>& get_cached_launchers()
{
  static std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>> launchers;
  return launchers;
}
