/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Suppress the inline `extract_code` definition in searcher_gpu_common.cuh;
// device_functions.cuh provides the extern declaration and the body lives in
// the extract_code JIT-LTO fragment.
#define IVF_RABITQ_JIT_LTO_FRAGMENT
#include "../gpu_index/searcher_gpu_common.cuh"
#include "device_functions.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

#include <cstdint>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Unified non-BlockSort LUT kernel body.
// WithEx=true: IP2 precomputed with long codes for all cluster vectors before the distance loop.
// WithEx=false: direct 1-bit LUT distance, no IP2 step.
template <bool WithEx>
__device__ void compute_inner_products_with_lut_impl(const ComputeInnerProductsKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  const size_t num_chunks         = params.D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  extern __shared__ __align__(256) char shared_mem_raw[];

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  if constexpr (WithEx) {
    // Shared memory layout: [ip2_results (max_candidates_per_pair floats)][lut / query (floats)]
    float* shared_ip2_results = reinterpret_cast<float*>(shared_mem_raw);
    float* shared_lut         = shared_ip2_results + params.max_candidates_per_pair;

    const size_t long_code_size = (params.D * params.ex_bits + 7) / 8;

    // Load query into the LUT region (reused; query fits since D ≤ lut_per_query_size)
    float* shared_query = shared_lut;
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    const int warp_id   = tid / raft::WarpSize;
    const int lane_id   = tid % raft::WarpSize;
    const int num_warps = num_threads / raft::WarpSize;

    for (int cand_idx = warp_id; cand_idx < num_vectors_in_cluster; cand_idx += num_warps) {
      size_t global_vec_idx        = cluster_start_index + cand_idx;
      const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

      float ip2 = 0.0f;
      for (size_t d = lane_id; d < params.D; d += raft::WarpSize) {
        uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits);
        float ex_val      = (float)code_val;
        ip2 += shared_query[d] * ex_val;
      }

#pragma unroll
      for (int offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
        ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
      }

      if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
    }
    __syncthreads();

    // Load LUT into the same region (overwrites query)
    float* query_lut = params.d_lut_for_queries_float + query_idx * lut_per_query_size;
    for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
      shared_lut[i] = query_lut[i];
    }

    const size_t short_code_length = params.D / 32;
    const size_t chunks_per_uint32 = 32 / BITS_PER_CHUNK;

    __shared__ int probe_slot;
    if (tid == 0) {
      probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
    }
    __syncthreads();

    float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    float q_kbxsumq = params.d_G_kbxSumq[query_idx];
    size_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
      size_t vec_idx = vec_base + tid;
      if (vec_idx >= num_vectors_in_cluster) break;

      float ip = 0.0f;
      for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
        size_t short_code_offset =
          cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
        uint32_t short_code_chunk = params.d_short_data[short_code_offset];
        for (int chunk_in_uint32 = 0; chunk_in_uint32 < chunks_per_uint32; chunk_in_uint32++) {
          int shift            = 28 - (chunk_in_uint32 * BITS_PER_CHUNK);
          int pattern          = (short_code_chunk >> shift) & 0xF;
          size_t lut_chunk_idx = uint32_idx * chunks_per_uint32 + chunk_in_uint32;
          size_t lut_offset    = lut_chunk_idx * LUT_SIZE + pattern;
          ip += shared_lut[lut_offset];
        }
      }

      float ip2             = shared_ip2_results[vec_idx];
      size_t global_vec_idx = cluster_start_index + vec_idx;

      float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
      float f_ex_add     = ex_factors.x;
      float f_ex_rescale = ex_factors.y;

      params.d_topk_dists[output_offset + vec_idx] =
        f_ex_add + q_g_add +
        f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);
      params.d_topk_pids[output_offset + vec_idx] = params.d_pids[global_vec_idx];
    }
  } else {
    // Shared memory layout: [lut (lut_per_query_size floats)]
    float* shared_lut = reinterpret_cast<float*>(shared_mem_raw);

    float* query_lut = params.d_lut_for_queries_float + query_idx * lut_per_query_size;
    for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
      shared_lut[i] = query_lut[i];
    }
    __syncthreads();

    float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    float q_k1xsumq = params.d_G_k1xSumq[query_idx];

    const size_t short_code_length = params.D / 32;
    const size_t chunks_per_uint32 = 32 / BITS_PER_CHUNK;

    __shared__ int probe_slot;
    if (tid == 0) {
      probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
    }
    __syncthreads();
    size_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
      size_t vec_idx = vec_base + tid;
      if (vec_idx >= num_vectors_in_cluster) break;

      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;

      float ip = 0.0f;
      for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
        size_t short_code_offset =
          cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
        uint32_t short_code_chunk = params.d_short_data[short_code_offset];
        for (int chunk_in_uint32 = 0; chunk_in_uint32 < chunks_per_uint32; chunk_in_uint32++) {
          int shift            = 28 - (chunk_in_uint32 * BITS_PER_CHUNK);
          int pattern          = (short_code_chunk >> shift) & 0xF;
          size_t lut_chunk_idx = uint32_idx * chunks_per_uint32 + chunk_in_uint32;
          size_t lut_offset    = lut_chunk_idx * LUT_SIZE + pattern;
          ip += shared_lut[lut_offset];
        }
      }

      params.d_topk_dists[output_offset + vec_idx] = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
      params.d_topk_pids[output_offset + vec_idx]  = params.d_pids[cluster_start_index + vec_idx];
    }
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
