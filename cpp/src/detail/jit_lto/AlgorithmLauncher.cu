/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>

#include <raft/util/cuda_rt_essentials.hpp>

AlgorithmLauncher::AlgorithmLauncher(cudaKernel_t k, cudaLibrary_t lib) : kernel{k}, library{lib} {}

AlgorithmLauncher::~AlgorithmLauncher()
{
  if (library != nullptr) { (void)cudaLibraryUnload(library); }
}

AlgorithmLauncher::AlgorithmLauncher(AlgorithmLauncher&& other) noexcept
  : kernel{other.kernel}, library{other.library}
{
  other.kernel  = nullptr;
  other.library = nullptr;
}

AlgorithmLauncher& AlgorithmLauncher::operator=(AlgorithmLauncher&& other) noexcept
{
  if (this != &other) {
    if (library != nullptr) { cudaLibraryUnload(library); }
    kernel        = other.kernel;
    library       = other.library;
    other.kernel  = nullptr;
    other.library = nullptr;
  }
  return *this;
}

void AlgorithmLauncher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  if (kernel == nullptr) { RAFT_FAIL("AlgorithmLauncher::call - kernel is NULL!"); }
  if (library == nullptr) { RAFT_FAIL("AlgorithmLauncher::call - library is NULL!"); }
  if (kernel_args == nullptr) { RAFT_FAIL("AlgorithmLauncher::call - kernel_args is NULL!"); }

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
