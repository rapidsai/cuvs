/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <driver_types.h>
#include <string>
#include <unordered_map>
#include <vector_types.h>

#include <cuda.h>
#include <memory>

struct AlgorithmLauncher {
  AlgorithmLauncher() : function{nullptr}, library{nullptr} {}

  AlgorithmLauncher(CUfunction f, CUlibrary lib);

  ~AlgorithmLauncher();

  AlgorithmLauncher(const AlgorithmLauncher&)            = delete;
  AlgorithmLauncher& operator=(const AlgorithmLauncher&) = delete;

  AlgorithmLauncher(AlgorithmLauncher&& other) noexcept;
  AlgorithmLauncher& operator=(AlgorithmLauncher&& other) noexcept;

  template <typename... Args>
  void dispatch(CUstream stream, dim3 grid, dim3 block, std::size_t shared_mem, Args&&... args)
  {
    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};
    this->call(stream, grid, block, shared_mem, kernel_args);
  }

  CUfunction get_function() { return this->function; }

 private:
  void call(CUstream stream, dim3 grid, dim3 block, std::size_t shared_mem, void** args);
  CUfunction function;
  CUlibrary library;
};

std::unordered_map<std::string, std::shared_ptr<AlgorithmLauncher>>& get_cached_launchers();
