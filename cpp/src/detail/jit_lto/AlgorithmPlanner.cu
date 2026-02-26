/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvjitlink_helper.hpp"

#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>

#include "cuda_runtime.h"

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

void AlgorithmPlanner::add_entrypoint()
{
  auto entrypoint_fragment = fragment_database().get_fragment(this->entrypoint);
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
    std::string log_message =
      "JIT compiling launcher for entrypoint: " + this->entrypoint + " and device functions: ";
    for (const auto& device_function : this->device_functions) {
      log_message += device_function + ",";
    }
    log_message.pop_back();
    RAFT_LOG_INFO("%s", log_message.c_str());
    launchers[launch_key] = this->build();
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
  const char* lopts[] = {"-lto", archs.c_str()};
  auto result         = nvJitLinkCreate(&handle, 2, lopts);
  check_nvjitlink_result(handle, result);

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
