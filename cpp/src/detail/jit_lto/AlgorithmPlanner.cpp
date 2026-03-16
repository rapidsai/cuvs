/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvjitlink_checker.hpp"

#include <chrono>
#include <iterator>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.hpp>
#include <cuvs/detail/jit_lto/FragmentDatabase.hpp>
#include <cuvs/detail/jit_lto/cu_try.hpp>

#include "cuda.h"
#include "cuda_runtime.h"
#include "nvJitLink.h"

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

void AlgorithmPlanner::add_entrypoint()
{
  auto entrypoint_fragment = fragment_database().get_fragment(this->fragment_key);
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
  auto launch_key = this->fragment_key + this->get_device_functions_key();

  static std::mutex cache_mutex;
  std::lock_guard<std::mutex> lock(cache_mutex);
  if (launchers.count(launch_key) == 0) {
    add_entrypoint();
    add_device_functions();
    std::string log_message =
      "JIT compiling launcher for kernel: " + this->fragment_key + " and device functions: ";
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
  CUdevice device;
  int major = 0;
  int minor = 0;
  CU_TRY(cuDeviceGet(&device, 0));
  CU_TRY(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CU_TRY(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

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
  CUlibrary library;
  CU_TRY(cuLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0));

  CUkernel kernel;
  CU_TRY(cuLibraryGetKernel(&kernel, library, this->entrypoint.c_str()));

  CUfunction function;
  CU_TRY(cuKernelGetFunction(&function, kernel));

  return std::make_shared<AlgorithmLauncher>(function, library);
}
