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

#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include <cuvs/detail/jit_lto/AlgorithmPlanner.h>
#include <cuvs/detail/jit_lto/FragmentDatabase.h>

#include "cuda.h"
#include "nvJitLink.h"

namespace {
// We can make a better RAII wrapper around nvjitlinkhandle
void check_nvjitlink_result(nvJitLinkHandle handle, nvJitLinkResult result)
{
  if (result != NVJITLINK_SUCCESS) {
    std::cerr << "\n nvJITLink failed with error " << result << '\n';
    size_t log_size = 0;
    result          = nvJitLinkGetErrorLogSize(handle, &log_size);
    if (result == NVJITLINK_SUCCESS && log_size > 0) {
      std::unique_ptr<char[]> log{new char[log_size]};
      result = nvJitLinkGetErrorLog(handle, log.get());
      if (result == NVJITLINK_SUCCESS) {
        std::cerr << "AlgorithmPlanner nvJITLink error log: " << log.get() << '\n';
      }
    }
    exit(1);
  }
}
}  // namespace

void AlgorithmPlanner::save_compute()
{
  std::cout << "Saving compute" << std::endl;
  auto& db = fragment_database();
  std::cout << "DB size: " << db.cache.size() << std::endl;
  std::cout << "Available keys in cache:" << std::endl;
  for (const auto& pair : db.cache) {
    std::cout << "  " << pair.first << std::endl;
  }
  std::cout << "Finding key: " << this->name + "_" + this->params << std::endl;
  auto val = db.cache.find(this->name + "_" + this->params);
  if (val == db.cache.end()) {
    std::cout << "Key not found" << std::endl;
    return;
  }
  this->fragments.push_back(val->second.get());
  std::cout << "Fragment added with key: " << fragments.back()->compute_key << std::endl;
  std::cout << "Fragments size: " << this->fragments.size() << std::endl;
}

AlgorithmLauncher AlgorithmPlanner::get_launcher()
{
  std::cout << "Getting launcher" << std::endl;
  auto& launchers = get_cached_launchers();
  auto key        = this->name + "_" + this->params;
  if (launchers.count(key) == 0) {
    this->save_compute();
    launchers[key] = this->build();
  }
  std::cout << "launcher key: " << key << std::endl;
  return launchers[key];
  // this->save_compute();
  // return this->build();
}

AlgorithmLauncher AlgorithmPlanner::build()
{
  int device = 0;
  int major  = 0;
  int minor  = 0;
  cudaGetDevice(&device);
  cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);

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
  std::cout << "\tStarted LTO runtime linking \n";
  result = nvJitLinkComplete(handle);
  check_nvjitlink_result(handle, result);
  std::cout << "\tCompleted LTO runtime linking \n";

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
  CUlibrary library;
  cuLibraryLoadData(&library, cubin.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);

  unsigned int count = 1;
  // Still need to cache/compute the mangled name
  std::unique_ptr<CUkernel[]> kernels_{new CUkernel[count]};
  cuLibraryEnumerateKernels(kernels_.get(), count, library);

  return AlgorithmLauncher{library, kernels_[0]};
}
