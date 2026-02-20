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
  RAFT_LOG_INFO("[JIT FRAGMENT] Looking up entrypoint fragment: %s", this->entrypoint.c_str());
  auto entrypoint_fragment = fragment_database().get_fragment(this->entrypoint);
  if (entrypoint_fragment == nullptr) {
    RAFT_FAIL("Entrypoint fragment is NULL for: %s", this->entrypoint.c_str());
  }
  RAFT_LOG_INFO("[JIT FRAGMENT] Entrypoint fragment found: %s (ptr: %p)",
                this->entrypoint.c_str(),
                entrypoint_fragment);
  this->fragments.push_back(entrypoint_fragment);
}

void AlgorithmPlanner::add_device_functions()
{
  for (const auto& device_function_key : this->device_functions) {
    RAFT_LOG_INFO("[JIT FRAGMENT] Looking up device function fragment: %s",
                  device_function_key.c_str());
    auto device_function_fragment = fragment_database().get_fragment(device_function_key);
    if (device_function_fragment == nullptr) {
      RAFT_FAIL("Device function fragment is NULL for: %s", device_function_key.c_str());
    }
    RAFT_LOG_INFO("[JIT FRAGMENT] Device function fragment found: %s (ptr: %p)",
                  device_function_key.c_str(),
                  device_function_fragment);
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
    RAFT_LOG_INFO(
      "[JIT CACHE] Cache MISS - Building new launcher for key: %s (entrypoint: %s, "
      "device_functions: %s)",
      launch_key.c_str(),
      this->entrypoint.c_str(),
      this->get_device_functions_key().c_str());
    add_entrypoint();
    add_device_functions();
    launchers[launch_key] = this->build();
    RAFT_LOG_INFO("[JIT CACHE] Launcher built and cached (kernel handle: %p)",
                  launchers[launch_key]->get_kernel());
  } else {
    RAFT_LOG_INFO(
      "[JIT CACHE] Cache HIT - Reusing cached launcher for key: %s (entrypoint: %s, kernel handle: "
      "%p)",
      launch_key.c_str(),
      this->entrypoint.c_str(),
      launchers[launch_key]->get_kernel());
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
  const char* lopts[] = {
    "-lto", "-split-compile=0", "-split-compile-extended=0", "-maxrregcount=64", archs.c_str()};
  auto result = nvJitLinkCreate(&handle, 5, lopts);
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

  // cubin is linked, so now load it
  cudaLibrary_t library;
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  unsigned int kernel_count = 0;
  RAFT_CUDA_TRY(cudaLibraryGetKernelCount(&kernel_count, library));

  // NOTE: cudaKernel_t does not need to be freed explicitly
  std::unique_ptr<cudaKernel_t[]> kernels{new cudaKernel_t[kernel_count]};
  RAFT_CUDA_TRY(cudaLibraryEnumerateKernels(kernels.get(), kernel_count, library));

  // Filter out EmptyKernel by checking kernel names using cudaFuncGetName
  const char* empty_kernel_name = "_ZN3cub6detail11EmptyKernelIvEEvv";
  std::vector<cudaKernel_t> valid_kernels;
  valid_kernels.reserve(kernel_count);

  for (unsigned int i = 0; i < kernel_count; ++i) {
    // cudaFuncGetName can be used with cudaKernel_t by casting to void*
    const void* func_ptr  = reinterpret_cast<const void*>(kernels[i]);
    const char* func_name = nullptr;
    RAFT_CUDA_TRY(cudaFuncGetName(&func_name, func_ptr));

    bool is_empty_kernel = false;
    if (func_name != nullptr) {
      std::string kernel_name(func_name);
      // Check if this is EmptyKernel
      if (kernel_name.find(empty_kernel_name) != std::string::npos ||
          kernel_name == empty_kernel_name) {
        is_empty_kernel = true;
      }
    }

    // Only keep the kernel if it's not EmptyKernel
    if (!is_empty_kernel) { valid_kernels.push_back(kernels[i]); }
  }

  RAFT_EXPECTS(
    valid_kernels.size() == 1, "Expected 1 valid JIT kernel, got %zu", valid_kernels.size());

  return std::make_shared<AlgorithmLauncher>(valid_kernels[0], library);
}
