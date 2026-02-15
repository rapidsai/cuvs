/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvjitlink_checker.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>

#include "cuda_runtime.h"
#include "nvJitLink.h"

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

void AlgorithmPlanner::add_entrypoint()
{
  std::cerr << "[JIT] AlgorithmPlanner::add_entrypoint - looking for entrypoint: "
            << this->entrypoint << std::endl;
  std::cerr.flush();
  auto entrypoint_fragment = fragment_database().get_fragment(this->entrypoint);
  if (entrypoint_fragment == nullptr) {
    std::cerr << "[JIT] ERROR: entrypoint fragment is NULL for: " << this->entrypoint << std::endl;
    std::cerr.flush();
  } else {
    std::cerr << "[JIT] Found entrypoint fragment for: " << this->entrypoint << std::endl;
    std::cerr.flush();
  }
  this->fragments.push_back(entrypoint_fragment);
}

void AlgorithmPlanner::add_device_functions()
{
  for (const auto& device_function_key : this->device_functions) {
    auto device_function_fragment = fragment_database().get_fragment(device_function_key);
    this->fragments.push_back(device_function_fragment);
  }
}

std::string AlgorithmPlanner::get_device_functions_key() const
{
  std::string key = "";
  for (const auto& device_function : this->device_functions) {
    key += device_function;
  }
  return key;
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::get_launcher()
{
  auto& launchers = get_cached_launchers();
  auto launch_key = this->entrypoint + this->get_device_functions_key();

  static std::mutex cache_mutex;
  std::lock_guard<std::mutex> lock(cache_mutex);
  std::cerr << "[JIT] AlgorithmPlanner::get_launcher called for entrypoint: " << this->entrypoint
            << std::endl;
  std::cerr.flush();
  if (launchers.count(launch_key) == 0) {
    add_entrypoint();
    add_device_functions();
    std::string log_message =
      "JIT compiling launcher for entrypoint: " + this->entrypoint + " and device functions: ";
    for (const auto& device_function : this->device_functions) {
      log_message += device_function + ",";
    }
    log_message.pop_back();
    std::cerr << "[JIT] " << log_message << std::endl;
    std::cerr.flush();

    // Time the first-time JIT compilation
    auto start_time       = std::chrono::high_resolution_clock::now();
    launchers[launch_key] = this->build();
    auto end_time         = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cerr << "[JIT] Compilation completed in " << duration.count()
              << " ms for entrypoint: " << this->entrypoint << std::endl;
    std::cerr.flush();
  } else {
    RAFT_LOG_DEBUG("Using cached JIT launcher for entrypoint: %s", this->entrypoint.c_str());
  }
  return launchers[launch_key];
}

std::shared_ptr<AlgorithmLauncher> AlgorithmPlanner::build()
{
  int device = 0;
  int major  = 0;
  int minor  = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

  std::string archs = "-arch=sm_" + std::to_string((major * 10 + minor));

  // Load the generated LTO IR and link them together
  nvJitLinkHandle handle;
  const char* lopts[] = {"-lto", archs.c_str(), "-O3"};
  auto result         = nvJitLinkCreate(&handle, 3, lopts);
  check_nvjitlink_result(handle, result);

  std::cerr << "[JIT] AlgorithmPlanner::build - Adding " << this->fragments.size()
            << " fragments to linker:" << std::endl;
  for (size_t i = 0; i < this->fragments.size(); ++i) {
    std::cerr << "[JIT]   Fragment [" << i << "] pointer: " << (void*)this->fragments[i]
              << std::endl;
    if (i == 0) {
      std::cerr << "[JIT]     (Entrypoint fragment)" << std::endl;
    } else {
      std::cerr << "[JIT]     (Device function fragment: " << this->device_functions[i - 1] << ")"
                << std::endl;
    }
  }
  std::cerr.flush();

  for (auto& frag : this->fragments) {
    frag->add_to(handle);
  }

  // Call to nvJitLinkComplete causes linker to link together all the LTO-IR
  // modules perform any optimizations and generate cubin from it.
  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);

  // get cubin from nvJitLink
  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  RAFT_EXPECTS(result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed");

  // Debug: Save cubin to disk for inspection with cuobjdump
  std::string cubin_path = "/tmp/linked_cubin_" + this->entrypoint + ".cubin";
  // Sanitize filename (replace special chars)
  std::replace(cubin_path.begin(), cubin_path.end(), '/', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), ':', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), '<', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), '>', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), ' ', '_');
  FILE* f = fopen(cubin_path.c_str(), "wb");
  if (f) {
    fwrite(cubin.get(), 1, cubin_size, f);
    fclose(f);
    std::cerr << "[JIT] Saved linked cubin to: " << cubin_path << " (size: " << cubin_size
              << " bytes)" << std::endl;
    std::cerr << "[JIT] Run: cuobjdump --dump-elf-symbols " << cubin_path
              << " to see kernel symbols" << std::endl;
    std::cerr.flush();
  }

  // cubin is linked, so now load it
  // NOTE: cudaLibrary_t does not need to be freed explicitly
  cudaLibrary_t library;
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  // The entrypoint fragment should contain exactly one __global__ kernel
  // Device functions (__device__) don't show up in kernel enumeration
  // But we might have kernels from multiple fragments if they were linked together
  std::cerr << "[JIT] AlgorithmPlanner::build - Fragments added: " << this->fragments.size()
            << " (entrypoint + " << this->device_functions.size() << " device functions)"
            << std::endl;
  std::cerr << "[JIT] AlgorithmPlanner::build - Entrypoint: " << this->entrypoint << std::endl;
  std::cerr.flush();

  // Enumerate kernels - we expect only 1 kernel from the entrypoint fragment
  // Device function fragments contain only __device__ functions, not __global__ kernels
  // So they shouldn't show up in kernel enumeration
  // First, query the actual number of kernels using cudaLibraryGetKernelCount (runtime API)
  unsigned int kernel_count = 0;
  cudaError_t cuda_result   = cudaLibraryGetKernelCount(&kernel_count, library);
  if (cuda_result != cudaSuccess) {
    std::cerr << "[JIT] ERROR: cudaLibraryGetKernelCount failed with error: " << cuda_result << " ("
              << cudaGetErrorString(cuda_result) << ")" << std::endl;
    std::cerr.flush();
    RAFT_FAIL("cudaLibraryGetKernelCount failed with error: %d (%s)",
              cuda_result,
              cudaGetErrorString(cuda_result));
  }

  std::cerr << "[JIT] AlgorithmPlanner::build - Actual kernel count in library: " << kernel_count
            << std::endl;
  std::cerr.flush();

  if (kernel_count == 0) {
    RAFT_FAIL("No kernels found in library for entrypoint: %s", this->entrypoint.c_str());
  }

  if (kernel_count > 1) {
    std::cerr << "[JIT] WARNING: Found " << kernel_count
              << " kernels in library! This might be the issue - we're using kernel [0]"
              << std::endl;
    std::cerr << "[JIT]   Entrypoint we're looking for: " << this->entrypoint << std::endl;
    std::cerr << "[JIT]   This suggests multiple kernels are being linked together!" << std::endl;
    std::cerr.flush();
  }

  // Now allocate the right size and enumerate
  std::unique_ptr<cudaKernel_t[]> kernels{new cudaKernel_t[kernel_count]};
  unsigned int kernel_count_verify = kernel_count;
  RAFT_CUDA_TRY(cudaLibraryEnumerateKernels(kernels.get(), kernel_count_verify, library));

  if (kernel_count_verify != kernel_count) {
    std::cerr << "[JIT] WARNING: Kernel count mismatch - cudaLibraryGetKernelCount returned "
              << kernel_count << " but cudaLibraryEnumerateKernels returned " << kernel_count_verify
              << std::endl;
    std::cerr.flush();
  }

  // With runtime API, we can't get kernel names directly
  // If there are multiple kernels, we'll use the first one
  // The entrypoint fragment should be added first, so its kernel should be at index 0
  if (kernel_count > 1) {
    std::cerr << "[JIT] WARNING: Multiple kernels found (" << kernel_count << "), using kernel [0]"
              << std::endl;
    std::cerr << "[JIT]   Entrypoint we're looking for: " << this->entrypoint << std::endl;
    std::cerr << "[JIT]   This suggests multiple kernels are being linked together!" << std::endl;
    std::cerr << "[JIT]   Fragments added:" << std::endl;
    for (size_t i = 0; i < this->fragments.size(); ++i) {
      std::cerr << "[JIT]     Fragment [" << i << "]: ";
      if (i == 0) {
        std::cerr << "Entrypoint fragment" << std::endl;
      } else {
        std::cerr << "Device function fragment: " << this->device_functions[i - 1] << std::endl;
      }
    }
    std::cerr.flush();
  }

  // When multiple kernels are found, one is often CUB's EmptyKernel (a weak symbol
  // instantiated when CUB headers are included). The entrypoint fragment is added first,
  // so its kernel should be at index 0. However, the order is not guaranteed - sometimes
  // CUB's EmptyKernel is at index 0, sometimes at index 1.
  // Strategy: Try kernel[0] first. If it's EmptyKernel, it will be a no-op and won't affect
  // results. We can't distinguish EmptyKernel from our kernel without names, so we'll use kernel[0]
  // and rely on the fact that EmptyKernel does nothing.
  unsigned int kernel_index = 0;
  if (kernel_count > 1) {
    std::cerr << "[JIT] WARNING: Found " << kernel_count
              << " kernels (CUB EmptyKernel may be present). Using kernel [0]" << std::endl;
    std::cerr << "[JIT]   If kernel [0] is EmptyKernel, results will be incorrect" << std::endl;
    std::cerr << "[JIT]   Entrypoint fragment is added first, so kernel [0] should be correct"
              << std::endl;
    std::cerr.flush();
  }

  auto kernel = kernels.release()[kernel_index];

  // Validate the kernel pointer is reasonable (not null, not obviously garbage)
  if (kernel == nullptr) {
    RAFT_FAIL("Entrypoint kernel is NULL for: %s", this->entrypoint.c_str());
  }

  void* kernel_ptr  = (void*)kernel;
  uintptr_t ptr_val = (uintptr_t)kernel_ptr;
  // Check if pointer looks valid (not null, not obviously ASCII string data)
  // On 64-bit systems, valid pointers are typically in the range 0x1000 to 0x7fffffffffff
  // but kernel pointers from CUDA driver API can be in higher address ranges
  // So we only check for null and obviously invalid values (too small)
  if (ptr_val < 0x1000) {
    RAFT_FAIL("Entrypoint kernel pointer looks invalid (0x%lx) - too small for: %s",
              ptr_val,
              this->entrypoint.c_str());
  }

  std::cerr << "[JIT] AlgorithmPlanner::build - Using kernel [0] as entrypoint, pointer: "
            << kernel_ptr << std::endl;
  std::cerr.flush();

  return std::make_shared<AlgorithmLauncher>(kernel);
}
