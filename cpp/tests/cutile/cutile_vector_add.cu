/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include "vector_add_kernel_symbol.h"
#include "vector_add_sm_100_cubin.h"
#include "vector_add_sm_120_cubin.h"
#include "vector_add_sm_80_cubin.h"
#include "vector_add_sm_86_cubin.h"
#include "vector_add_sm_90_cubin.h"
#include "vector_add_tileir_bytecode.h"

#include <cuvs/detail/jit_lto/tileir_compat.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace cuvs {
namespace {

struct EmbeddedCubin {
  int cc_major;
  int cc_minor;
  const unsigned char* data;
  size_t size;
};

// Prebuilt cubins for known library targets (see export_vector_add_cubin.py).
constexpr EmbeddedCubin kEmbeddedCubins[] = {
  {8, 0, vector_add_sm_80_cubin, sizeof(vector_add_sm_80_cubin)},
  {8, 6, vector_add_sm_86_cubin, sizeof(vector_add_sm_86_cubin)},
  {9, 0, vector_add_sm_90_cubin, sizeof(vector_add_sm_90_cubin)},
  {10, 0, vector_add_sm_100_cubin, sizeof(vector_add_sm_100_cubin)},
  {12, 0, vector_add_sm_120_cubin, sizeof(vector_add_sm_120_cubin)},
};

constexpr EmbeddedCubin kTileIrBytecode = {
  -1,
  -1,
  vector_add_tileir_bytecode,
  sizeof(vector_add_tileir_bytecode),
};

struct CutileModuleImage {
  const uint8_t* data;
  size_t size;
};

std::optional<CutileModuleImage> resolve_vector_add_module(int cc_major, int cc_minor)
{
  for (const auto& entry : kEmbeddedCubins) {
    if (entry.cc_major == cc_major && entry.cc_minor == cc_minor) {
      return CutileModuleImage{reinterpret_cast<const uint8_t*>(entry.data), entry.size};
    }
  }

  int driver_version = 0;
  if (cudaDriverGetVersion(&driver_version) != cudaSuccess) { return std::nullopt; }
  if (!cuvs::detail::jit_lto::tileir_fallback_available(driver_version)) {
    return std::nullopt;
  }
  return CutileModuleImage{
    reinterpret_cast<const uint8_t*>(kTileIrBytecode.data), kTileIrBytecode.size};
}

struct LoadedKernel {
  cudaLibrary_t library = nullptr;
  cudaKernel_t kernel     = nullptr;
  bool used_tileir_jit{false};
  const char* skip_reason{nullptr};

  LoadedKernel() = default;

  LoadedKernel(LoadedKernel&& other) noexcept { *this = std::move(other); }

  LoadedKernel& operator=(LoadedKernel&& other) noexcept
  {
    if (this != &other) {
      unload();
      library         = other.library;
      kernel          = other.kernel;
      used_tileir_jit = other.used_tileir_jit;
      skip_reason     = other.skip_reason;
      other.library   = nullptr;
      other.kernel    = nullptr;
    }
    return *this;
  }

  LoadedKernel(const LoadedKernel&)            = delete;
  LoadedKernel& operator=(const LoadedKernel&) = delete;

  ~LoadedKernel() { unload(); }

  explicit operator bool() const { return kernel != nullptr; }

 private:
  void unload()
  {
    if (library != nullptr) {
      RAFT_CUDA_TRY(cudaLibraryUnload(library));
      library = nullptr;
      kernel  = nullptr;
    }
  }
};

LoadedKernel load_vector_add_kernel(int cc_major, int cc_minor)
{
  LoadedKernel result{};
  result.used_tileir_jit = !cuvs::detail::jit_lto::is_embedded_cubin_arch(cc_major, cc_minor);

  auto image = resolve_vector_add_module(cc_major, cc_minor);
  if (!image) {
    if (result.used_tileir_jit) {
      result.skip_reason =
        "TileIR driver JIT unavailable for this GPU. Requires CUDA 13.1+ driver (>= 590.44).";
    } else {
      ADD_FAILURE() << "No embedded cuTile module for compute capability " << cc_major << "."
                    << cc_minor;
    }
    return result;
  }

  const cudaError_t load_status =
    cudaLibraryLoadData(&result.library, image->data, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (load_status != cudaSuccess) {
    if (result.used_tileir_jit) {
      result.skip_reason =
        "TileIR driver JIT unavailable for this GPU (requires CUDA 13.1+ driver >= 590.44).";
      SCOPED_TRACE(cudaGetErrorString(load_status));
    } else {
      ADD_FAILURE() << "cudaLibraryLoadData failed: " << cudaGetErrorString(load_status);
    }
    return result;
  }

  const cudaError_t kernel_status =
    cudaLibraryGetKernel(&result.kernel, result.library, CUTILE_VECTOR_ADD_KERNEL_SYMBOL);
  if (kernel_status != cudaSuccess) {
    if (result.library != nullptr) {
      RAFT_CUDA_TRY(cudaLibraryUnload(result.library));
      result.library = nullptr;
    }
    result.kernel = nullptr;
    if (result.used_tileir_jit) {
      result.skip_reason =
        "TileIR driver JIT unavailable for this GPU (requires CUDA 13.1+ driver >= 590.44).";
      SCOPED_TRACE(cudaGetErrorString(kernel_status));
    } else {
      ADD_FAILURE() << "cudaLibraryGetKernel failed: " << cudaGetErrorString(kernel_status);
    }
  }
  return result;
}

void run_vector_add(cudaKernel_t kernel)
{
  constexpr int kN       = 1024;
  constexpr int kTile    = 256;
  constexpr int kGridDim = (kN + kTile - 1) / kTile;

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(&d_a, kN * sizeof(float)));
  RAFT_CUDA_TRY(cudaMalloc(&d_b, kN * sizeof(float)));
  RAFT_CUDA_TRY(cudaMalloc(&d_c, kN * sizeof(float)));

  std::vector<float> h_a(kN), h_b(kN);
  for (int i = 0; i < kN; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }
  RAFT_CUDA_TRY(cudaMemcpy(d_a, h_a.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  RAFT_CUDA_TRY(cudaMemcpy(d_b, h_b.data(), kN * sizeof(float), cudaMemcpyHostToDevice));
  RAFT_CUDA_TRY(cudaMemset(d_c, 0, kN * sizeof(float)));

  int64_t shape  = kN;
  int64_t stride = 1;
  void* kernel_args[] = {
    &d_a, &shape, &stride, &d_b, &shape, &stride, &d_c, &shape, &stride,
  };

  dim3 grid(kGridDim);
  dim3 block(1);
  ASSERT_EQ(cudaSuccess, cudaLaunchKernel(kernel, grid, block, kernel_args, 0, 0))
    << "cudaLaunchKernel failed: " << cudaGetErrorString(cudaGetLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::vector<float> h_c(kN);
  RAFT_CUDA_TRY(cudaMemcpy(h_c.data(), d_c, kN * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < kN; ++i) {
    ASSERT_FLOAT_EQ(h_a[i] + h_b[i], h_c[i]) << "@" << i;
  }

  RAFT_CUDA_TRY(cudaFree(d_a));
  RAFT_CUDA_TRY(cudaFree(d_b));
  RAFT_CUDA_TRY(cudaFree(d_c));
}

class CutileVectorAddTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int device = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device));
    RAFT_CUDA_TRY(
      cudaDeviceGetAttribute(&cc_major_, cudaDevAttrComputeCapabilityMajor, device));
    RAFT_CUDA_TRY(
      cudaDeviceGetAttribute(&cc_minor_, cudaDevAttrComputeCapabilityMinor, device));
  }

  int cc_major_{};
  int cc_minor_{};
};

}  // namespace

TEST_F(CutileVectorAddTest, EmbeddedCubinVectorAdd)
{
  LoadedKernel loaded = load_vector_add_kernel(cc_major_, cc_minor_);
  if (loaded.skip_reason) { GTEST_SKIP() << loaded.skip_reason; }
  if (!loaded) { return; }

  SCOPED_TRACE(loaded.used_tileir_jit ? "loaded via TileIR driver JIT"
                                      : "loaded via prebuilt cubin");
  run_vector_add(loaded.kernel);
}

}  // namespace cuvs
