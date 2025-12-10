/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#include "searcher_gpu.cuh"

#include <raft/matrix/select_k.cuh>

#include <cstdint>
#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_rabitq::detail {
namespace {

#define MAX_TOP_K               64  // power of 2, as local_topk_capacity, assumes that topk is less than 100
#define MAX_CANDIDATES_PER_PAIR 1000  // suppose topk = 100, M = 10

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16
static constexpr int WARP_SIZE      = 32;

// --- Tunables ---
using T    = float;
using IdxT = uint32_t;

using lut_dtype = __half;  // FP16 alternative

// function to extract long codes
__device__ inline uint32_t extract_code(const uint8_t* codes, size_t d, size_t EX_BITS)
{
  size_t bitPos    = d * EX_BITS;
  size_t byteIdx   = bitPos >> 3;
  size_t bitOffset = bitPos & 7;
  uint32_t v       = codes[byteIdx] << 8;
  if (bitOffset + EX_BITS > 8) { v |= codes[byteIdx + 1]; }
  int shift = 16 - (bitOffset + EX_BITS);
  return (v >> shift) & ((1u << EX_BITS) - 1);
}

// Kernel to init invalid distances
__global__ void initDistancesKernel(float* d_input_dists, size_t total_elements)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total_elements) { d_input_dists[tid] = INFINITY; }
}

// Final function to merge top-k results from all clusters
void mergeClusterTopKFinal(const float* d_topk_dists,  // Input: top-k distances from all clusters
                           const PID* d_topk_pids,     // Input: top-k PIDs from all clusters
                           float* d_final_dists,  // Output: final top-k distances for each query
                           PID* d_final_pids,     // Output: final top-k PIDs for each query
                           size_t num_queries,
                           size_t nprobe,
                           size_t topk,
                           raft::resources const& handle,
                           bool sorted = false  // Whether to sort the final results
)
{
  auto stream = raft::resource::get_cuda_stream(handle);

  size_t candidates_per_query = nprobe * topk;

  raft::matrix::detail::select_k(handle,
                                 d_topk_dists,
                                 d_topk_pids,
                                 num_queries,
                                 candidates_per_query,
                                 topk,
                                 d_final_dists,
                                 d_final_pids,
                                 true,
                                 sorted);
}

}  // namespace
}  // namespace cuvs::neighbors::ivf_rabitq::detail
