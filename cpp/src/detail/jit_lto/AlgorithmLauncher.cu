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

#include <raft/util/cuda_rt_essentials.hpp>

AlgorithmLauncher::AlgorithmLauncher(cudaKernel_t k) : kernel{k} {}

void AlgorithmLauncher::call(
  cudaStream_t stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t config;
  config.gridDim          = grid;
  config.blockDim         = block;
  config.stream           = stream;
  config.attrs            = attribute;
  config.numAttrs         = 1;
  config.dynamicSmemBytes = shared_mem;

  RAFT_CUDA_TRY(cudaLaunchKernelExC(&config, kernel, kernel_args));
}

std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>>& get_cached_launchers()
{
  static std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>> launchers;
  return launchers;
}
