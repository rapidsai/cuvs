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

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cuvs {
namespace {

struct EmbeddedCubin {
  int cc_major;
  int cc_minor;
  const unsigned char* data;
  size_t size;
};

// Lookup table for cubins built at configure time (see export_vector_add_cubin.py).
constexpr EmbeddedCubin kEmbeddedCubins[] = {
  {8, 0, vector_add_sm_80_cubin, sizeof(vector_add_sm_80_cubin)},
  {8, 6, vector_add_sm_86_cubin, sizeof(vector_add_sm_86_cubin)},
  {9, 0, vector_add_sm_90_cubin, sizeof(vector_add_sm_90_cubin)},
  {10, 0, vector_add_sm_100_cubin, sizeof(vector_add_sm_100_cubin)},
  {12, 0, vector_add_sm_120_cubin, sizeof(vector_add_sm_120_cubin)},
};

const EmbeddedCubin* find_embedded_cubin(int cc_major, int cc_minor)
{
  for (const auto& entry : kEmbeddedCubins) {
    if (entry.cc_major == cc_major && entry.cc_minor == cc_minor) { return &entry; }
  }
  // Fall back to a cubin for the same major version (e.g. minor SKUs within a generation).
  for (const auto& entry : kEmbeddedCubins) {
    if (entry.cc_major == cc_major) { return &entry; }
  }
  return nullptr;
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
  const EmbeddedCubin* cubin = find_embedded_cubin(cc_major_, cc_minor_);
  ASSERT_NE(cubin, nullptr)
    << "No embedded cuTile cubin for compute capability " << cc_major_ << "." << cc_minor_;

  cudaLibrary_t library{};
  ASSERT_EQ(cudaSuccess,
            cudaLibraryLoadData(
              &library, cubin->data, nullptr, nullptr, 0, nullptr, nullptr, 0))
    << "cudaLibraryLoadData failed: " << cudaGetErrorString(cudaGetLastError());

  cudaKernel_t kernel{};
  ASSERT_EQ(cudaSuccess,
            cudaLibraryGetKernel(&kernel, library, CUTILE_VECTOR_ADD_KERNEL_SYMBOL))
    << "cudaLibraryGetKernel failed: " << cudaGetErrorString(cudaGetLastError());

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
  RAFT_CUDA_TRY(cudaLibraryUnload(library));
}

}  // namespace cuvs
