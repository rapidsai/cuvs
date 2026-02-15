/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvjitlink_checker.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>

#include "cuda_runtime.h"
#include "nvJitLink.h"

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

void AlgorithmPlanner::add_entrypoint()
{
  auto entrypoint_fragment = fragment_database().get_fragment(this->entrypoint);
  if (entrypoint_fragment == nullptr) {
    RAFT_FAIL("Entrypoint fragment is NULL for: %s", this->entrypoint.c_str());
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
  if (launchers.count(launch_key) == 0) {
    add_entrypoint();
    add_device_functions();
    launchers[launch_key] = this->build();
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

  for (auto& frag : this->fragments) {
    frag->add_to(handle);
  }

  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);

  size_t cubin_size;
  result = nvJitLinkGetLinkedCubinSize(handle, &cubin_size);
  check_nvjitlink_result(handle, result);

  std::unique_ptr<char[]> cubin{new char[cubin_size]};
  result = nvJitLinkGetLinkedCubin(handle, cubin.get());
  check_nvjitlink_result(handle, result);

  result = nvJitLinkDestroy(&handle);
  RAFT_EXPECTS(result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed");

  // Save cubin to disk for inspection with cuobjdump
  std::string cubin_path = "/tmp/linked_cubin_" + this->entrypoint + ".cubin";
  // Sanitize filename (replace special chars)
  std::replace(cubin_path.begin(), cubin_path.end(), '/', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), ':', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), '<', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), '>', '_');
  std::replace(cubin_path.begin(), cubin_path.end(), ' ', '_');
  FILE* f = fopen(cubin_path.c_str(), "wb");
  if (f) {
    size_t written = fwrite(cubin.get(), 1, cubin_size, f);
    fclose(f);
    if (written == cubin_size) {
      std::cerr << "[JIT] =========================================" << std::endl;
      std::cerr << "[JIT] Saved linked cubin to: " << cubin_path << " (size: " << cubin_size
                << " bytes)" << std::endl;
      std::cerr << "[JIT] Run: cuobjdump --dump-elf-symbols " << cubin_path
                << " to see kernel symbols" << std::endl;
      std::cerr << "[JIT] =========================================" << std::endl;
      std::cerr.flush();
    } else {
      std::cerr << "[JIT] WARNING: Failed to write full cubin (wrote " << written << " of "
                << cubin_size << " bytes)" << std::endl;
      std::cerr.flush();
    }
  } else {
    std::cerr << "[JIT] WARNING: Failed to open cubin file for writing: " << cubin_path
              << " (errno: " << errno << ")" << std::endl;
    std::cerr.flush();
  }

  // cubin is linked, so now load it
  cudaLibrary_t library;
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  // Enumerate kernels
  unsigned int kernel_count = 0;
  cudaError_t cuda_result   = cudaLibraryGetKernelCount(&kernel_count, library);
  if (cuda_result != cudaSuccess) {
    RAFT_FAIL("cudaLibraryGetKernelCount failed with error: %d (%s)",
              cuda_result,
              cudaGetErrorString(cuda_result));
  }

  std::cerr << "[JIT] Kernel count in library: " << kernel_count << std::endl;
  std::cerr.flush();

  if (kernel_count == 0) {
    RAFT_FAIL("No kernels found in library for entrypoint: %s", this->entrypoint.c_str());
  }

  if (kernel_count > 1) {
    std::cerr << "[JIT] WARNING: Found " << kernel_count << " kernels in library! Using kernel [0]"
              << std::endl;
    std::cerr.flush();
  }

  std::unique_ptr<cudaKernel_t[]> kernels{new cudaKernel_t[kernel_count]};
  unsigned int kernel_count_verify = kernel_count;
  RAFT_CUDA_TRY(cudaLibraryEnumerateKernels(kernels.get(), kernel_count_verify, library));

  if (kernel_count_verify != kernel_count) {
    RAFT_FAIL(
      "Kernel count mismatch: cudaLibraryGetKernelCount returned %u but "
      "cudaLibraryEnumerateKernels returned %u",
      kernel_count,
      kernel_count_verify);
  }

  unsigned int kernel_index = 0;

  auto kernel = kernels.release()[kernel_index];

  if (kernel == nullptr) {
    RAFT_FAIL("Entrypoint kernel is NULL for: %s", this->entrypoint.c_str());
  }

  return std::make_shared<AlgorithmLauncher>(kernel);
}
