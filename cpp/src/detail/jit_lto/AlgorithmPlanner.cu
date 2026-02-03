/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvjitlink_checker.hpp"

#include <chrono>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.h>
#include <cuvs/detail/jit_lto/FragmentDatabase.h>
#include <iostream>

#include "cuda_runtime.h"
#include "nvJitLink.h"

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
  if (launchers.count(launch_key) == 0) {
    add_entrypoint();
    add_device_functions();
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
  const char* lopts[] = {"-lto", archs.c_str(), "-O3"};
  auto result         = nvJitLinkCreate(&handle, 3, lopts);
  check_nvjitlink_result(handle, result);

  for (auto& frag : this->fragments) {
    std::cout << "Adding fragment: " << frag->compute_key << std::endl;
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
  check_nvjitlink_result(handle, result);

  // cubin is linked, so now load it
  // NOTE: cudaLibrary_t does not need to be freed explicitly
  cudaLibrary_t library;
  RAFT_CUDA_TRY(
    cudaLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  constexpr unsigned int count = 1;
  // NOTE: cudaKernel_t does not need to be freed explicitly
  std::unique_ptr<cudaKernel_t[]> kernels{new cudaKernel_t[count]};
  RAFT_CUDA_TRY(cudaLibraryEnumerateKernels(kernels.get(), count, library));

  return std::make_shared<AlgorithmLauncher>(kernels.release()[0]);
}
