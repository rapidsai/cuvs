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

#include <cuvs/detail/jit_lto/AlgorithmLauncher.h>

#include <iostream>

AlgorithmLauncher::AlgorithmLauncher(CUlibrary l, CUkernel k) : library{l}, kernel{k} {}

void AlgorithmLauncher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  CUlaunchAttribute attribute[1];
  attribute[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attribute[0].value.programmaticStreamSerializationAllowed = 1;

  CUlaunchConfig config{};
  config.gridDimX       = grid.x;
  config.gridDimY       = grid.y;
  config.gridDimZ       = grid.z;
  config.blockDimX      = block.x;
  config.blockDimY      = block.y;
  config.blockDimZ      = block.z;
  config.sharedMemBytes = shared_mem;
  config.hStream        = stream;
  config.attrs          = attribute;
  config.numAttrs       = 1;

  cuLaunchKernelEx(&config, (CUfunction)kernel, kernel_args, 0);
}

std::unordered_map<std::string, AlgorithmLauncher>& get_cached_launchers()
{
  static std::unordered_map<std::string, AlgorithmLauncher> launchers;
  return launchers;
}
