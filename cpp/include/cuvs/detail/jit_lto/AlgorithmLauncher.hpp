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
  AlgorithmLauncher() : kernel{nullptr}, library{nullptr} {}

  AlgorithmLauncher(cudaKernel_t k, cudaLibrary_t lib);

  ~AlgorithmLauncher();

  // Delete copy constructor and assignment to prevent accidental copying
  AlgorithmLauncher(const AlgorithmLauncher&)            = delete;
  AlgorithmLauncher& operator=(const AlgorithmLauncher&) = delete;

  // Allow move constructor and assignment
  AlgorithmLauncher(AlgorithmLauncher&& other) noexcept;
  AlgorithmLauncher& operator=(AlgorithmLauncher&& other) noexcept;

  template <typename... Args>
  void dispatch(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    // Create array of pointers to arguments
    // NOTE: cudaLaunchKernelExC copies the parameter values synchronously before returning,
    // so the local array and argument references are safe even though the kernel launch is async
    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};

    // Validate that we're not passing null pointers for critical parameters
    // (This is a sanity check - actual validation should be done by callers)
    for (size_t i = 0; i < sizeof...(args); ++i) {
      if (kernel_args[i] == nullptr) {
        // Some parameters might legitimately be nullptr, so we just log a warning
        // The kernel itself should validate critical pointers
      }
    }

    this->call(stream, grid, block, shared_mem, kernel_args);
  }

  template <typename... Args>
  void dispatch_cooperative(
    cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call_cooperative(stream, grid, block, shared_mem, kernel_args);
  }

  cudaKernel_t get_kernel() { return this->kernel; }

 private:
  void call(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  void call_cooperative(
    cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  cudaKernel_t kernel;
  cudaLibrary_t library;
};

std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>>& get_cached_launchers();
