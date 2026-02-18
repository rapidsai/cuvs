/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>

#include <iostream>

#include <raft/util/cuda_rt_essentials.hpp>

AlgorithmLauncher::AlgorithmLauncher(cudaKernel_t k, cudaLibrary_t lib) : kernel{k}, library{lib} {}

AlgorithmLauncher::~AlgorithmLauncher()
{
  if (library != nullptr) {
    cudaError_t err = cudaLibraryUnload(library);
    if (err != cudaSuccess) {
      // Log error but don't throw in destructor
      std::cerr << "[JIT] WARNING: Failed to unload library in destructor: "
                << cudaGetErrorString(err) << std::endl;
      std::cerr.flush();
    }
    library = nullptr;
  }
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
    // Unload current library if it exists
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
  // Validate kernel and library handles before use
  if (kernel == nullptr) { RAFT_FAIL("AlgorithmLauncher::call - kernel is NULL!"); }
  if (library == nullptr) { RAFT_FAIL("AlgorithmLauncher::call - library is NULL!"); }
  if (kernel_args == nullptr) { RAFT_FAIL("AlgorithmLauncher::call - kernel_args is NULL!"); }

  // Debug: verify kernel is being called
  std::cerr << "[JIT] AlgorithmLauncher::call - kernel is not null, launching with grid=(" << grid.x
            << "," << grid.y << "," << grid.z << ") block=(" << block.x << "," << block.y << ","
            << block.z << ")" << std::endl;
  std::cerr.flush();

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

  // NOTE: cudaLaunchKernelExC copies parameter values synchronously before returning,
  // so the kernel_args array and the values it points to are safe even though the launch is async
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

  // Check for immediate errors after launch (catches parameter issues early)
  cudaError_t peek_err = cudaPeekAtLastError();
  if (peek_err != cudaSuccess) {
    std::cerr << "[JIT] WARNING: Error detected immediately after kernel launch: "
              << cudaGetErrorString(peek_err) << " (" << peek_err << ")" << std::endl;
    std::cerr.flush();
  }
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
