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

  // Generate individual cubin for each device function fragment for debugging
  // Skip entrypoint fragment as it depends on device functions and will fail to link alone
  // for (auto& frag : this->fragments) {
  //   // Skip if this is the entrypoint fragment
  //   if (frag->compute_key == this->entrypoint) { continue; }

  //   nvJitLinkHandle frag_handle;
  //   const char* frag_lopts[] = {"-lto", archs.c_str()};
  //   auto frag_result         = nvJitLinkCreate(&frag_handle, 2, frag_lopts);
  //   check_nvjitlink_result(frag_handle, frag_result);

  //   frag->add_to(frag_handle);

  //   frag_result = nvJitLinkComplete(frag_handle);
  //   check_nvjitlink_result(frag_handle, frag_result);

  //   size_t frag_cubin_size;
  //   frag_result = nvJitLinkGetLinkedCubinSize(frag_handle, &frag_cubin_size);
  //   check_nvjitlink_result(frag_handle, frag_result);

  //   if (frag_cubin_size > 0) {
  //     std::unique_ptr<char[]> frag_cubin{new char[frag_cubin_size]};
  //     frag_result = nvJitLinkGetLinkedCubin(frag_handle, frag_cubin.get());
  //     check_nvjitlink_result(frag_handle, frag_result);

  //     // Save individual fragment cubin
  //     std::string frag_cubin_path = "/tmp/fragment_cubin_" + frag->compute_key + ".cubin";
  //     std::replace(frag_cubin_path.begin(), frag_cubin_path.end(), '/', '_');
  //     std::replace(frag_cubin_path.begin(), frag_cubin_path.end(), ':', '_');
  //     std::replace(frag_cubin_path.begin(), frag_cubin_path.end(), '<', '_');
  //     std::replace(frag_cubin_path.begin(), frag_cubin_path.end(), '>', '_');
  //     std::replace(frag_cubin_path.begin(), frag_cubin_path.end(), ' ', '_');
  //     FILE* frag_f = fopen(frag_cubin_path.c_str(), "wb");
  //     if (frag_f) {
  //       size_t written = fwrite(frag_cubin.get(), 1, frag_cubin_size, frag_f);
  //       fclose(frag_f);
  //       if (written == frag_cubin_size) {
  //         std::cerr << "[JIT] Saved fragment cubin: " << frag_cubin_path
  //                   << " (size: " << frag_cubin_size << " bytes)" << std::endl;
  //         std::cerr << "[JIT] Run: cuobjdump --dump-elf-symbols " << frag_cubin_path <<
  //         std::endl;
  //       }
  //     }
  //   }

  //   frag_result = nvJitLinkDestroy(&frag_handle);
  //   RAFT_EXPECTS(frag_result == NVJITLINK_SUCCESS, "nvJitLinkDestroy failed for fragment");
  // }

  // Load the generated LTO IR and link them together
  nvJitLinkHandle handle;
  const char* lopts[] = {"-lto", archs.c_str()};
  auto result         = nvJitLinkCreate(&handle, 2, lopts);
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
  // std::string cubin_path = "/tmp/linked_cubin_" + this->entrypoint + ".cubin";
  // // Sanitize filename (replace special chars)
  // std::replace(cubin_path.begin(), cubin_path.end(), '/', '_');
  // std::replace(cubin_path.begin(), cubin_path.end(), ':', '_');
  // std::replace(cubin_path.begin(), cubin_path.end(), '<', '_');
  // std::replace(cubin_path.begin(), cubin_path.end(), '>', '_');
  // std::replace(cubin_path.begin(), cubin_path.end(), ' ', '_');
  // FILE* f = fopen(cubin_path.c_str(), "wb");
  // if (f) {
  //   size_t written = fwrite(cubin.get(), 1, cubin_size, f);
  //   fclose(f);
  //   if (written == cubin_size) {
  //     std::cerr << "[JIT] =========================================" << std::endl;
  //     std::cerr << "[JIT] Saved linked cubin to: " << cubin_path << " (size: " << cubin_size
  //               << " bytes)" << std::endl;
  //     std::cerr << "[JIT] Run: cuobjdump --dump-elf-symbols " << cubin_path
  //               << " to see kernel symbols" << std::endl;
  //     std::cerr << "[JIT] =========================================" << std::endl;
  //     std::cerr.flush();
  //   } else {
  //     std::cerr << "[JIT] WARNING: Failed to write full cubin (wrote " << written << " of "
  //               << cubin_size << " bytes)" << std::endl;
  //     std::cerr.flush();
  //   }
  // } else {
  //   std::cerr << "[JIT] WARNING: Failed to open cubin file for writing: " << cubin_path
  //             << " (errno: " << errno << ")" << std::endl;
  //   std::cerr.flush();
  // }

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

  // Filter out EmptyKernel by checking kernel names using cudaFuncGetName
  const char* empty_kernel_name = "_ZN3cub6detail11EmptyKernelIvEEvv";
  std::vector<cudaKernel_t> valid_kernels;
  valid_kernels.reserve(kernel_count);

  for (unsigned int i = 0; i < kernel_count; ++i) {
    // cudaFuncGetName can be used with cudaKernel_t by casting to void*
    const void* func_ptr    = reinterpret_cast<const void*>(kernels[i]);
    const char* func_name   = nullptr;
    cudaError_t name_result = cudaFuncGetName(&func_name, func_ptr);

    bool is_empty_kernel = false;
    if (name_result == cudaSuccess && func_name != nullptr) {
      std::string kernel_name(func_name);
      // Check if this is EmptyKernel
      if (kernel_name.find(empty_kernel_name) != std::string::npos ||
          kernel_name == empty_kernel_name) {
        std::cerr << "[JIT] Filtering out EmptyKernel: " << kernel_name << std::endl;
        std::cerr.flush();
        is_empty_kernel = true;
      } else {
        std::cerr << "[JIT] Found kernel: " << kernel_name << std::endl;
        std::cerr.flush();
      }
    } else {
      // If we can't get the name, keep the kernel (better safe than sorry)
      std::cerr << "[JIT] Warning: Could not get name for kernel [" << i
                << "], keeping it (error: " << cudaGetErrorString(name_result) << ")" << std::endl;
      std::cerr.flush();
    }

    // Only keep the kernel if it's not EmptyKernel
    if (!is_empty_kernel) { valid_kernels.push_back(kernels[i]); }
  }

  if (valid_kernels.empty()) {
    RAFT_FAIL("No valid kernels found after filtering EmptyKernel for entrypoint: %s",
              this->entrypoint.c_str());
  }

  if (valid_kernels.size() > 1) {
    std::cerr << "[JIT] WARNING: Found " << valid_kernels.size()
              << " valid kernels after filtering! Using kernel [0]" << std::endl;
    std::cerr.flush();
  }

  unsigned int kernel_index = 0;
  auto kernel               = valid_kernels[kernel_index];

  if (kernel == nullptr) {
    RAFT_FAIL("Entrypoint kernel is NULL for: %s", this->entrypoint.c_str());
  }

  return std::make_shared<AlgorithmLauncher>(kernel, library);
}
