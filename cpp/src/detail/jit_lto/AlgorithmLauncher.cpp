/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/detail/jit_lto/AlgorithmLauncher.hpp>
#include <cuvs/detail/jit_lto/cu_try.hpp>

AlgorithmLauncher::AlgorithmLauncher(CUfunction f, CUlibrary lib) : function{f}, library{lib} {}

AlgorithmLauncher::~AlgorithmLauncher()
{
  if (library != nullptr) { (void)cuLibraryUnload(library); }
}

AlgorithmLauncher::AlgorithmLauncher(AlgorithmLauncher&& other) noexcept
  : function{other.function}, library{other.library}
{
  other.function = nullptr;
  other.library  = nullptr;
}

AlgorithmLauncher& AlgorithmLauncher::operator=(AlgorithmLauncher&& other) noexcept
{
  if (this != &other) {
    // Unload current library if it exists
    if (library != nullptr) { cuLibraryUnload(library); }
    function       = other.function;
    library        = other.library;
    other.function = nullptr;
    other.library  = nullptr;
  }
  return *this;
}

void AlgorithmLauncher::call(
  CUstream stream, dim3 grid, dim3 block, std::size_t shared_mem, void** kernel_args)
{
  CUlaunchAttribute attribute[1];
  attribute[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
  attribute[0].value.programmaticStreamSerializationAllowed = 1;

  CUlaunchConfig config;
  config.gridDimX       = grid.x;
  config.gridDimY       = grid.y;
  config.gridDimZ       = grid.z;
  config.blockDimX      = block.x;
  config.blockDimY      = block.y;
  config.blockDimZ      = block.z;
  config.hStream        = stream;
  config.attrs          = attribute;
  config.numAttrs       = 1;
  config.sharedMemBytes = shared_mem;

  CU_TRY(cuLaunchKernelEx(&config, function, kernel_args, nullptr));
}

std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>>& get_cached_launchers()
{
  static std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>> launchers;
  return launchers;
}
