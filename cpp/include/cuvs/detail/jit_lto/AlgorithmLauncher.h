/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <driver_types.h>
#include <string>
#include <unordered_map>
#include <vector_types.h>

#include <cuda_runtime.h>
#include <memory>

struct AlgorithmLauncher {
  AlgorithmLauncher() = default;

  AlgorithmLauncher(cudaKernel_t k);

  template <typename... Args>
  void dispatch(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call(stream, grid, block, shared_mem, kernel_args);
  }

  cudaKernel_t get_kernel() { return this->kernel; }

 private:
  void call(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  cudaKernel_t kernel;
};

std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>>& get_cached_launchers();
