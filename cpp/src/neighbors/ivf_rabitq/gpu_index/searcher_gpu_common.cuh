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

#define MAX_TOP_K_BLOCK_SORT 64     // power of 2; increases shared mem usage
#define MAX_TOP_K_WARP_SORT  16384  // power of 2; does not increase shared mem usage
#define WARP_SORT_CAPACITY_FACTOR \
  4                // factor between successive capacity values for instantiating warp_sort class
#define STR(x) #x  // convert macro value to literal string

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16
static constexpr int WARP_SIZE      = 32;

// --- Tunables ---
using T    = float;
using IdxT = uint32_t;

using lut_dtype = __half;  // FP16 alternative

// POD struct consolidating parameters for all computeInnerProducts* kernels
struct ComputeInnerProductsKernelParams {
  const ClusterQueryPair* d_sorted_pairs       = nullptr;
  const float* d_query                         = nullptr;
  const uint32_t* d_short_data                 = nullptr;
  const IVFGPU::GPUClusterMeta* d_cluster_meta = nullptr;
  float* d_lut_for_queries_float               = nullptr;
  lut_dtype* d_lut_for_queries_half            = nullptr;
  const uint32_t* d_packed_queries             = nullptr;  // Packed query bit planes
  const float* d_widths                        = nullptr;  // Query scaling factors
  const float* d_short_factors                 = nullptr;
  const float* d_G_k1xSumq                     = nullptr;
  const float* d_G_kbxSumq                     = nullptr;
  const float* d_centroid_distances            = nullptr;
  uint32_t topk                                = 0;
  uint32_t num_queries                         = 0;
  uint32_t nprobe                              = 0;
  uint32_t num_pairs                           = 0;
  uint32_t num_centroids                       = 0;
  uint32_t D                                   = 0;
  const float* d_threshold                     = nullptr;  // threshold for each query
  uint32_t max_candidates_per_pair             = 0;        // max storage per pair, 1000 suggested
  uint32_t ex_bits                             = 0;        // bits per dimension in ex codes
  const uint8_t* d_long_code                   = nullptr;  // long codes for all vectors
  const float* d_ex_factor                     = nullptr;  // ex factors for distance computation
  const PID* d_pids                            = nullptr;  // PIDs for all vectors
  float* d_topk_dists                          = nullptr;  // output top-k distances
  PID* d_topk_pids                             = nullptr;  // output top-k PIDs
  int* d_query_write_counters                  = nullptr;
  uint32_t num_bits                            = 0;  // number of bits (8 for int8)
  uint32_t num_words                           = 0;  // approx. D/32
};

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
