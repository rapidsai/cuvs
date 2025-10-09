/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <driver_types.h>
#include <string>
#include <unordered_map>
#include <vector_types.h>

#include <cuda_runtime.h>
#include <nvJitLink.h>

struct AlgorithmLauncher {
  AlgorithmLauncher() = default;

  AlgorithmLauncher(cudaLibrary_t l, cudaKernel_t k);

  template <typename... Args>
  void operator()(
    cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call(stream, grid, block, shared_mem, kernel_args);
  }

  cudaKernel_t get_kernel() { return this->kernel; }

 private:
  void call(cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  cudaLibrary_t library;
  cudaKernel_t kernel;
};

std::unordered_map<std::string, AlgorithmLauncher>& get_cached_launchers();
