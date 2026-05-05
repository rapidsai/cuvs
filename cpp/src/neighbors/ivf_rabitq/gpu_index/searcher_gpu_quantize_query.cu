/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

// This file implements `SearcherGPU::SearchClusterQueryPairsQuantizeQuery`.
#include "../../detail/smem_utils.cuh"
#include "../../ivf_flat/detail/jit_lto_kernels/interleaved_scan_impl.cuh"
#include "../utils/searcher_gpu_utils.hpp"
#include "searcher_gpu.cuh"
#include "searcher_gpu_common.cuh"

#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select_k.cuh>

#include <cub/block/block_reduce.cuh>

#include <thrust/fill.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace cuvs::neighbors::ivf_rabitq::detail {

__global__ void computeInnerProductsWithBitwiseOpt(const ComputeInnerProductsKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Allocate shared memory for candidates
  float* shared_query       = reinterpret_cast<float*>(shared_mem_raw_2);
  float* shared_ip2_results = shared_query + params.D;

  for (size_t i = tid; i < params.D; i += num_threads) {
    shared_query[i] = params.d_query[query_idx * params.D + i];
  }
  __syncthreads();

  // Step 1: Warp-level IP2 computation for better memory coalescing
  const int warp_id   = tid / raft::WarpSize;
  const int lane_id   = tid % raft::WarpSize;
  const int num_warps = num_threads / raft::WarpSize;

  // Calculate long code parameters
  const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

  // Each warp processes different candidates
  for (int cand_idx = warp_id; cand_idx < num_vectors_in_cluster; cand_idx += num_warps) {
    size_t global_vec_idx = cluster_start_index + cand_idx;

    // Pointer to this vector's long code
    const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

    // Warp-level IP2 computation
    float ip2 = 0.0f;

    // Each thread in warp processes different dimensions
    for (uint32_t d = lane_id; d < params.D; d += raft::WarpSize) {
      // Extract ex_bits value for this dimension
      uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits);
      float ex_val      = (float)code_val;
      ip2 += shared_query[d] * ex_val;
    }

    // Warp-level reduction for ip2
#pragma unroll
    for (int offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
      ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
    }

    // Lane 0 stores the result
    if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
  }

  __syncthreads();

  // Step 2: Load float query and compute exact IPs for candidates

  const size_t short_code_length = params.D / 32;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_kbxsumq = params.d_G_kbxSumq[query_idx];

  // Atomically get write position
  __shared__ int probe_slot;
  if (tid == 0) {
    probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
  }
  __syncthreads();
  // Calculate output offset
  uint32_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx >= num_vectors_in_cluster) break;

    // Compute exact inner product with float query
    float exact_ip = 0.0f;

    // Process each uint32_t of the short code
    for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
      // Access short code in transposed layout
      size_t short_code_offset =
        cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
      uint32_t short_code_chunk = params.d_short_data[short_code_offset];

      // Process each bit in the uint32_t
      // Note: bit 31 is lowest dimension, bit 0 is highest
#pragma unroll 8
      for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
        size_t dim = uint32_idx * 32 + bit_idx;
        if (dim < params.D) {
          // Extract bit from MSB to LSB
          int bit_position = 31 - bit_idx;
          bool bit_value   = (short_code_chunk >> bit_position) & 0x1;

          // If bit is 1, add the query value
          if (bit_value) { exact_ip += shared_query[dim]; }
        }
      }
    }

    // Get pre-computed values
    float ip2             = shared_ip2_results[vec_idx];
    size_t global_vec_idx = cluster_start_index + vec_idx;

    // vec load version
    float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
    float f_ex_add     = ex_factors.x;
    float f_ex_rescale = ex_factors.y;

    // Compute final distance using pre-computed ip2
    float ex_dist =
      f_ex_add + q_g_add +
      f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);

    // Write to global memory
    params.d_topk_dists[output_offset + vec_idx] = ex_dist;
    params.d_topk_pids[output_offset + vec_idx]  = params.d_pids[global_vec_idx];
  }
}

__global__ void computeInnerProductsWithBitwiseOptNoEX(
  const ComputeInnerProductsKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw[];

  // Allocate shared memory for candidates
  float* shared_query            = reinterpret_cast<float*>(shared_mem_raw);
  const size_t short_code_length = params.D / 32;

  for (size_t i = tid; i < params.D; i += num_threads) {
    shared_query[i] = params.d_query[query_idx * params.D + i];
  }
  __syncthreads();

  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_k1xsumq = params.d_G_k1xSumq[query_idx];

  // Step 2: Load float query and compute exact IPs for candidates

  // Compute exact float inner products for all candidates
  // Atomically get write position
  __shared__ int probe_slot;
  if (tid == 0) {
    probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
  }
  __syncthreads();
  // Calculate output offset
  uint32_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx >= num_vectors_in_cluster) break;

    size_t factor_offset  = cluster_start_index + vec_idx;
    float3 factors        = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
    float f_add           = factors.x;
    float f_rescale       = factors.y;
    size_t global_vec_idx = cluster_start_index + vec_idx;

    // Compute exact inner product with float query
    float exact_ip = 0.0f;

    // Process each uint32_t of the short code
    for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
      // Access short code in transposed layout
      size_t short_code_offset =
        cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
      uint32_t short_code_chunk = params.d_short_data[short_code_offset];

      // Process each bit in the uint32_t
      // Note: bit 31 is lowest dimension, bit 0 is highest
#pragma unroll 8
      for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
        size_t dim = uint32_idx * 32 + bit_idx;
        if (dim < params.D) {
          // Extract bit from MSB to LSB
          int bit_position = 31 - bit_idx;
          bool bit_value   = (short_code_chunk >> bit_position) & 0x1;

          // If bit is 1, add the query value
          if (bit_value) { exact_ip += shared_query[dim]; }
        }
      }
    }

    // get final results
    float final_1bit_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
    PID final_1bit_pid    = (uint32_t)params.d_pids[global_vec_idx];
    // Write to global memory
    params.d_topk_dists[output_offset + vec_idx] = final_1bit_dist;
    params.d_topk_pids[output_offset + vec_idx]  = final_1bit_pid;
  }
}

// Unified template replacing the four separate BlockSort kernels.
// NumBits=4 or 8; WithEx=true adds warp-level IP2 refinement with long codes.
template <int NumBits, bool WithEx>
__global__ void computeInnerProductsWithBitwiseBlockSort(
  const ComputeInnerProductsKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  extern __shared__ __align__(256) char shared_mem_raw_2[];
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  float query_width = params.d_widths[query_idx];

  __shared__ int num_candidates;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_k1xsumq = params.d_G_k1xSumq[query_idx];
  float q_g_error = sqrtf(q_g_add);
  float threshold = params.d_threshold[query_idx];

  if (tid == 0) { num_candidates = 0; }
  __syncthreads();

  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;

  // Phase 1: Bitwise inner product filter
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate        = false;
    float local_ip_quantized = 0;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = 0;
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];

#pragma unroll
        for (int b = 0; b < NumBits - 1; b++) {
          accumulator += __popc(shared_packed_query[b * params.num_words + word] & data_word) << b;
        }
        accumulator -=
          __popc(shared_packed_query[(NumBits - 1) * params.num_words + word] & data_word)
          << (NumBits - 1);
      }

      float ip       = (float)accumulator * query_width;
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
      float low_dist = est_dist - f_error * q_g_error;

      if (low_dist < threshold) {
        is_candidate       = true;
        local_ip_quantized = ip;
      }
    }

    __syncwarp();

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = local_ip_quantized;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }

  __syncthreads();

  if (num_candidates > 0) {
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;
    __shared__ int probe_slot;

    if constexpr (WithEx) {
      // Phase 2 (WithEx): Compute exact 1-bit IPs and store for IP2 refinement
      for (int c = 0; c < candidates_per_thread; ++c) {
        int cand_idx = tid + c * num_threads;
        if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
          int vec_idx    = shared_candidate_indices[cand_idx];
          float exact_ip = 0.0f;

          for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
            size_t short_code_offset = cluster_start_index * short_code_length +
                                       uint32_idx * num_vectors_in_cluster + vec_idx;
            uint32_t short_code_chunk = params.d_short_data[short_code_offset];
#pragma unroll 8
            for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
              size_t dim = uint32_idx * 32 + bit_idx;
              if (dim < params.D) {
                if ((short_code_chunk >> (31 - bit_idx)) & 0x1) { exact_ip += shared_query[dim]; }
              }
            }
          }
          shared_candidate_ips[cand_idx] = exact_ip;
        }
      }
      __syncthreads();

      // Phase 3 (WithEx): Warp-level IP2 computation + block-sort queue
      {
        using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
          flat_block_sort<kMaxTopKBlockSort, true, T, IdxT>::type;
        block_sort_t queue(params.topk);

        float q_kbxsumq               = params.d_G_kbxSumq[query_idx];
        const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;
        float* shared_ip2_results     = reinterpret_cast<float*>(shared_mem_raw_2);

        const int warp_id   = tid / raft::WarpSize;
        const int lane_id   = tid % raft::WarpSize;
        const int num_warps = num_threads / raft::WarpSize;

        for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
          size_t global_vec_idx        = cluster_start_index + shared_candidate_indices[cand_idx];
          const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

          float ip2 = 0.0f;
          for (uint32_t d = lane_id; d < params.D; d += raft::WarpSize) {
            ip2 += shared_query[d] * (float)extract_code(vec_long_code, d, params.ex_bits);
          }
#pragma unroll
          for (int offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
            ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
          }
          if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
        }
        __syncthreads();

        for (int round = 0; round < candidates_per_thread; round++) {
          int cand_idx = tid + round * num_threads;

          float ex_dist;
          uint32_t pid;
          if (cand_idx < num_candidates) {
            float ip              = shared_candidate_ips[cand_idx];
            float ip2             = shared_ip2_results[cand_idx];
            int local_vec_idx     = shared_candidate_indices[cand_idx];
            size_t global_vec_idx = cluster_start_index + local_vec_idx;

            float2 ex_factors = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
            float f_ex_add    = ex_factors.x;
            float f_ex_rescale = ex_factors.y;

            ex_dist =
              f_ex_add + q_g_add +
              f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);
            pid = (uint32_t)params.d_pids[global_vec_idx];
          } else {
            ex_dist = INFINITY;
            pid     = 0;
          }
          queue.add(ex_dist, pid);
        }
        __syncthreads();

        queue.done((uint8_t*)shared_mem_raw_2);
        if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
        __syncthreads();

        if (probe_slot >= params.nprobe) { return; }

        uint32_t output_offset =
          query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
        queue.store(params.d_topk_dists + output_offset,
                    (uint32_t*)(params.d_topk_pids + output_offset));
      }
    } else {
      // Phase 2+3 (NoEx): Compute exact 1-bit IPs and add directly to queue
      using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
        flat_block_sort<kMaxTopKBlockSort, true, T, IdxT>::type;
      block_sort_t queue(params.topk);

      float final_dist;
      PID final_pid;
      for (int c = 0; c < candidates_per_thread; ++c) {
        int cand_idx = tid + c * num_threads;
        if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
          int vec_idx          = shared_candidate_indices[cand_idx];
          size_t factor_offset = cluster_start_index + vec_idx;
          float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
          float f_add     = factors.x;
          float f_rescale = factors.y;
          size_t global_vec_idx = cluster_start_index + vec_idx;

          float exact_ip = 0.0f;
          for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
            size_t short_code_offset = cluster_start_index * short_code_length +
                                       uint32_idx * num_vectors_in_cluster + vec_idx;
            uint32_t short_code_chunk = params.d_short_data[short_code_offset];
#pragma unroll 8
            for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
              size_t dim = uint32_idx * 32 + bit_idx;
              if (dim < params.D) {
                if ((short_code_chunk >> (31 - bit_idx)) & 0x1) { exact_ip += shared_query[dim]; }
              }
            }
          }
          final_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
          final_pid  = (uint32_t)params.d_pids[global_vec_idx];
        } else {
          final_dist = INFINITY;
          final_pid  = 0;
        }
        queue.add(final_dist, final_pid);
      }
      __syncthreads();

      queue.done((uint8_t*)shared_mem_raw_2);
      if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
      __syncthreads();

      uint32_t output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
      queue.store(params.d_topk_dists + output_offset,
                  (uint32_t*)(params.d_topk_pids + output_offset));
    }

    // Update threshold atomically
    if (num_candidates >= params.topk) {
      float max_topk_dist;

      if (tid == 0) {
        max_topk_dist = -INFINITY;
        uint32_t output_offset =
          query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
        for (uint32_t i = 0; i < params.topk; i++) {
          float dist = params.d_topk_dists[output_offset + i];
          if (dist > 0 && dist > max_topk_dist && dist < INFINITY) { max_topk_dist = dist; }
        }
      }
      __syncthreads();

      if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
        atomicMin((int*)(params.d_threshold + query_idx), __float_as_int(max_topk_dist));
      }
    }
  }
}

__inline__ __device__ float warpReduceSum(float v)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__inline__ __device__ float blockReduceSum(float v)
{
  __shared__ float shared[32];  // up to 1024 threads -> 32 warps
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;

  v = warpReduceSum(v);
  if (lane == 0) shared[wid] = v;
  __syncthreads();

  float out = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.f;
  if (wid == 0) out = warpReduceSum(out);
  return out;
}

//---------------------------------------------------------------------------
// Kernel: exrabitq_quantize_query
//
// Quantize queries using exrabitq implementation, the output are always int8_t array
//
template <unsigned int BlockSize>
__global__ void exrabitq_quantize_query(
  // Inputs
  const float* __restrict__ d_XP,
  size_t num_points,
  size_t D,
  size_t EX_BITS,
  float const_scaling_factor,
  float kConstEpsilon,
  // Outputs
  int8_t* d_long_code,
  float* d_delta)
{
  //=========================================================================
  // Setup: One block per row
  //=========================================================================
  int row = blockIdx.x;
  if (row >= num_points) return;

  // Dynamically allocated shared memory for one row's data.
  extern __shared__ float s_mem[];
  float* s_xp        = s_mem;
  int8_t* s_tmp_code = (int8_t*)(s_xp + D);
  float* s_reduce    = (float*)(s_tmp_code + D);  // For reduction

  int tid = threadIdx.x;

  //=========================================================================
  // Step 0: Load XP and compute L2 nrom && normalize
  //=========================================================================
  float thread_sum_sq = 0.0f;

  // local L2 norm
  for (int j = tid; j < D; j += BlockSize) {
    float xp_val = d_XP[row * D + j];
    s_xp[j]      = xp_val;
    thread_sum_sq += xp_val * xp_val;  // Direct L2 norm of XP
  }

  s_reduce[tid] = thread_sum_sq;
  __syncthreads();

  // global reduction
  for (unsigned int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) { s_reduce[tid] += s_reduce[tid + stride]; }
    __syncthreads();
  }

  float norm     = sqrtf(s_reduce[0]);
  float norm_inv = (norm > 0) ? (1.0f / norm) : 0.0f;

  //=========================================================================
  // Step 1 (skipped): Coalesced load of all necessary data into shared memory
  //=========================================================================

  //=========================================================================
  // Part A: ExRaBitQ Code Generation
  //=========================================================================
  // Parallel quantization and start of ip_norm reduction
  for (int j = tid; j < D; j += BlockSize) {
    float val    = s_xp[j] * norm_inv;
    int code_val = __float2int_rn((const_scaling_factor * val) /*+ 0.5*/);  // round-to-nearest-even
    if (code_val > (1 << (EX_BITS - 1)) - 1) code_val = (1 << (EX_BITS - 1)) - 1;
    if (code_val < (-(1 << (EX_BITS - 1)))) code_val = -(1 << (EX_BITS - 1));
    s_tmp_code[j] = code_val;
  }
  __syncthreads();

  //=========================================================================
  // Part B: Factor Computation
  //=========================================================================
  float ip_resi_xucb = 0.f, xu_sq = 0.f;

  for (size_t j = tid; j < D; j += BlockSize) {
    float res  = s_xp[j];
    int xu_pre = s_tmp_code[j];

    float xu = float(xu_pre) /* - (static_cast<float>(1 << (EX_BITS - 1))) */;
    // just ignore the 0.5 since we are not going to store extra shift
    ip_resi_xucb += res * xu;  // for cos_similarity
    xu_sq += xu * xu;          // norm_quan^2
  }

  // only thread 0 in the block need the results, so simply use blockReduceSum
  // Perform parallel reductions for all factor components
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  xu_sq        = blockReduceSum(xu_sq);

  // Thread 0 computes and writes the final factors
  if (tid == 0) {
    float norm_quan      = sqrtf(fmaxf(xu_sq, 0.f));
    float cos_similarity = ip_resi_xucb / (norm * norm_quan);
    float delta          = norm / norm_quan * cos_similarity;

    size_t base   = row;
    d_delta[base] = delta;
  }

  //=========================================================================
  // Part C: Pack and Write Long Code (MINIMAL READS, PARALLEL, COALESCED)
  //=========================================================================
  int long_code_length = D;  // D dims, then D bytes
  int8_t* out_ptr      = d_long_code + row * long_code_length;
  for (int j = tid; j < D; j += BlockSize)
    out_ptr[j] = s_tmp_code[j];  // write outputs directly
}

__global__ void findQueryRanges(const float* __restrict__ queries,
                                float* __restrict__ query_ranges,
                                int num_queries,
                                int num_dimensions)
{
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const float* query = queries + query_idx * num_dimensions;

  using BlockReduceFloat = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduceFloat::TempStorage temp_storage_min;
  __shared__ typename BlockReduceFloat::TempStorage temp_storage_max;

  float local_min = FLT_MAX;
  float local_max = -FLT_MAX;

  for (int i = threadIdx.x; i < num_dimensions; i += blockDim.x) {
    float val = query[i];
    local_min = fminf(local_min, val);
    local_max = fmaxf(local_max, val);
  }

  float block_min = BlockReduceFloat(temp_storage_min).Reduce(local_min, cuda::minimum<>{});
  __syncthreads();
  float block_max = BlockReduceFloat(temp_storage_max).Reduce(local_max, cuda::maximum<>{});

  if (threadIdx.x == 0) {
    query_ranges[query_idx * 2]     = block_min;
    query_ranges[query_idx * 2 + 1] = block_max;
  }
}

__global__ void quantizeQueriesToInt8(const float* __restrict__ queries,
                                      const float* __restrict__ query_ranges,
                                      int8_t* __restrict__ quantized_queries,
                                      float* __restrict__ widths,
                                      int num_queries,
                                      int num_dimensions)
{
  const int BQ            = 8;                    // Use full 8-bit range
  const float max_int_val = (1 << (BQ - 1)) - 1;  // 127

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_dimensions;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / num_dimensions;
    int dim_idx   = i % num_dimensions;

    float vmin     = query_ranges[query_idx * 2];
    float vmax     = query_ranges[query_idx * 2 + 1];
    float vmax_abs = fmaxf(fabsf(vmin), fabsf(vmax));

    float width          = vmax_abs / max_int_val;
    float one_over_width = (width > 0) ? 1.0f / width : 0.0f;

    if (dim_idx == 0) { widths[query_idx] = width; }

    float val        = queries[query_idx * num_dimensions + dim_idx];
    float scaled     = val * one_over_width;
    scaled           = fmaxf(-128.0f, fminf(127.0f, scaled));
    int8_t quantized = (int8_t)__float2int_rn(scaled);

    quantized_queries[query_idx * num_dimensions + dim_idx] = quantized;
  }
}

__global__ void packInt8QueryBitPlanes(const int8_t* __restrict__ queries,
                                       uint32_t* __restrict__ packed_queries,
                                       int num_queries,
                                       int num_dimensions)
{
  const int dims_per_word = 32;
  const int num_words     = (num_dimensions + dims_per_word - 1) / dims_per_word;
  const int num_bits      = 8;

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_bits * num_words;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / (num_bits * num_words);
    int remainder = i % (num_bits * num_words);
    int bit_idx   = remainder / num_words;
    int word_idx  = remainder % num_words;

    uint32_t packed_word = 0;

#pragma unroll 8
    for (int d = 0; d < dims_per_word; ++d) {
      int dim_idx = word_idx * dims_per_word + d;
      if (dim_idx < num_dimensions) {
        uint8_t val      = (uint8_t)queries[query_idx * num_dimensions + dim_idx];
        uint32_t bit_val = (val >> bit_idx) & 1;

        // FIXED: Match data bit ordering - dim 0 goes to bit 31, dim 31 to bit 0
        int bit_position = 31 - d;  // Reverse bit ordering!
        packed_word |= (bit_val << bit_position);
      }
    }

    packed_queries[i] = packed_word;
  }
}

__global__ void quantizeQueriesToInt4(
  const float* __restrict__ queries,
  const float* __restrict__ query_ranges,
  int8_t* __restrict__ quantized_queries,  // Still use int8_t for storage
  float* __restrict__ widths,
  int num_queries,
  int num_dimensions)
{
  const int BQ            = 4;                    // Use 4-bit range
  const float max_int_val = (1 << (BQ - 1)) - 1;  // 2^3 - 1 = 7

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_dimensions;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / num_dimensions;
    int dim_idx   = i % num_dimensions;

    float vmin     = query_ranges[query_idx * 2];
    float vmax     = query_ranges[query_idx * 2 + 1];
    float vmax_abs = fmaxf(fabsf(vmin), fabsf(vmax));

    float width          = vmax_abs / max_int_val;
    float one_over_width = (width > 0) ? 1.0f / width : 0.0f;

    if (dim_idx == 0) { widths[query_idx] = width; }

    float val    = queries[query_idx * num_dimensions + dim_idx];
    float scaled = val * one_over_width;

    // Clamp to 4-bit range [-8, 7]
    scaled           = fmaxf(-8.0f, fminf(7.0f, scaled));
    int8_t quantized = (int8_t)__float2int_rn(scaled);

    quantized_queries[query_idx * num_dimensions + dim_idx] = quantized;
  }
}

__global__ void packInt4QueryBitPlanes(const int8_t* __restrict__ queries,
                                       uint32_t* __restrict__ packed_queries,
                                       int num_queries,
                                       int num_dimensions)
{
  const int dims_per_word = 32;
  const int num_words     = (num_dimensions + dims_per_word - 1) / dims_per_word;
  const int num_bits      = 4;  // Only 4 bit planes!

  int idx            = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = num_queries * num_bits * num_words;

  for (int i = idx; i < total_elements; i += gridDim.x * blockDim.x) {
    int query_idx = i / (num_bits * num_words);
    int remainder = i % (num_bits * num_words);
    int bit_idx   = remainder / num_words;
    int word_idx  = remainder % num_words;

    uint32_t packed_word = 0;

#pragma unroll 8
    for (int d = 0; d < dims_per_word; ++d) {
      int dim_idx = word_idx * dims_per_word + d;
      if (dim_idx < num_dimensions) {
        // For 4-bit values, we only care about the lower 4 bits
        // But need to handle sign extension properly
        uint8_t val      = (uint8_t)(queries[query_idx * num_dimensions + dim_idx] & 0xF);
        uint32_t bit_val = (val >> bit_idx) & 1;

        // Match data bit ordering - dim 0 goes to bit 31
        int bit_position = 31 - d;
        packed_word |= (bit_val << bit_position);
      }
    }

    packed_queries[i] = packed_word;
  }
}

// Search with qunatized query vectors
void SearcherGPU::SearchClusterQueryPairsQuantizeQuery(
  const IVFGPU& cur_ivf,
  IVFGPU::GPUClusterMeta* d_cluster_meta,
  ClusterQueryPair* d_sorted_pairs,
  size_t num_queries,
  const float* d_query,
  const float* d_G_k1xSumq,
  const float* d_G_kbxSumq,
  size_t nprobe,
  size_t topk,
  float* d_final_dists,
  PID* d_final_pids,
  bool use_4bit  // Add parameter to choose 4-bit or 8-bit
)
{
  // check if the inner products kernel should use block sort to keep a top-k priority queue vs.
  // outputting distances from all vectors in probed clusters
  const bool use_block_sort{topk <= kMaxTopKBlockSort};

  // query quantize
  const int num_bits  = use_4bit ? 4 : 8;  // Choose bit width
  const int num_words = (cur_ivf.get_num_padded_dim() + 31) / 32;

  // Allocate memory for quantization
  auto d_query_write_counters = raft::make_device_vector<int, int64_t>(handle_, num_queries);
  auto d_query_ranges         = raft::make_device_vector<float, int64_t>(handle_, 0);
  auto d_widths               = raft::make_device_vector<float, int64_t>(handle_, 0);
  auto d_quantized_queries    = raft::make_device_vector<int8_t, int64_t>(handle_, 0);
  auto d_packed_queries       = raft::make_device_vector<uint32_t, int64_t>(handle_, 0);
  auto d_topk_threshold_batch = raft::make_device_vector<float, int64_t>(handle_, 0);
  if (use_block_sort) {
    d_query_ranges      = raft::make_device_vector<float, int64_t>(handle_, num_queries * 2);
    d_widths            = raft::make_device_vector<float, int64_t>(handle_, num_queries);
    d_quantized_queries = raft::make_device_vector<int8_t, int64_t>(
      handle_, num_queries * cur_ivf.get_num_padded_dim());
    d_packed_queries =
      raft::make_device_vector<uint32_t, int64_t>(handle_, num_queries * num_bits * num_words);
    d_topk_threshold_batch = raft::make_device_vector<float, int64_t>(handle_, num_queries);
  }

  if (use_block_sort) {
    if (rabitq_quantize_flag_) {
      const int block_size = 256;
      const int grid_size  = num_queries;
      size_t shared_mem    = D * sizeof(float) + D * sizeof(int8_t) + block_size * sizeof(float);
      exrabitq_quantize_query<block_size>
        <<<grid_size, block_size, shared_mem, stream_>>>(d_query,
                                                         num_queries,
                                                         D,
                                                         num_bits,
                                                         best_rescaling_factor,
                                                         1.9f,
                                                         d_quantized_queries.data_handle(),
                                                         d_widths.data_handle());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {  // scalar quantize
      // Step 1: Find min/max for each query
      const int block_size = 256;
      const int grid_size  = num_queries;
      findQueryRanges<<<grid_size, block_size, 0, stream_>>>(
        d_query, d_query_ranges.data_handle(), num_queries, cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Step 2: Quantize queries to int8_t with BQ=8
      if (use_4bit) {
        quantizeQueriesToInt4<<<grid_size, block_size, 0, stream_>>>(
          d_query,
          d_query_ranges.data_handle(),
          d_quantized_queries.data_handle(),
          d_widths.data_handle(),
          num_queries,
          cur_ivf.get_num_padded_dim());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      } else {
        quantizeQueriesToInt8<<<grid_size, block_size, 0, stream_>>>(
          d_query,
          d_query_ranges.data_handle(),
          d_quantized_queries.data_handle(),
          d_widths.data_handle(),
          num_queries,
          cur_ivf.get_num_padded_dim());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    }
  }

  // Step 3: Pack quantized queries into bit planes
  if (use_block_sort) {
    const int block_size = 256;
    const int grid_size  = (num_queries * num_bits * num_words + block_size - 1) / block_size;

    if (use_4bit) {
      packInt4QueryBitPlanes<<<grid_size, block_size, 0, stream_>>>(
        d_quantized_queries.data_handle(),
        d_packed_queries.data_handle(),
        num_queries,
        cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {
      packInt8QueryBitPlanes<<<grid_size, block_size, 0, stream_>>>(
        d_quantized_queries.data_handle(),
        d_packed_queries.data_handle(),
        num_queries,
        cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }

  // We minimize max_cluster_size to reduce shared memory usage when the probe clusters do not
  // include the largest cluster. This optimization is expected to be more effective when
  // num_queries and/or nprobe are low.
  uint32_t max_cluster_size;

  // For the intermediate distances (and associated IDs), we want to minimize the allocation both to
  // reduce memory footprint and to avoid unnecessary passes in the final RAFT select_k call. For
  // `use_block_sort = true`, the required allocation is simply num_queries * nprobe * topk. For
  // `use_block_sort = false`, the strategy here is to compute the sum of cluster sizes over all
  // probed clusters for each query, and use the maximum of these sums as the allocation size needed
  // per query. This avoids negative performance impact from any abnormally large cluster when using
  // the global maximum cluster size as the allocation size per query per probe.
  std::optional<size_t> max_probed_vectors_count =
    use_block_sort ? std::nullopt : std::optional<size_t>{0};

  // call utility function to evaluate max_cluster_size and max_probed_vectors_count
  get_max_probed_cluster_size_and_vectors_count(handle_,
                                                d_sorted_pairs,
                                                num_queries * nprobe,
                                                cur_ivf.get_cluster_meta().data_handle(),
                                                num_queries,
                                                max_cluster_size,
                                                max_probed_vectors_count);

  // allocate memory for intermediate output
  size_t total_elements =
    use_block_sort ? num_queries * nprobe * topk : num_queries * max_probed_vectors_count.value();
  auto d_topk_dists = raft::make_device_vector<float, int64_t>(handle_, total_elements);
  auto d_topk_pids  = raft::make_device_vector<PID, int64_t>(handle_, total_elements);

  // initialize distances
  thrust::fill(thrust::cuda::par.on(stream_),
               d_topk_dists.data_handle(),
               d_topk_dists.data_handle() + total_elements,
               std::numeric_limits<float>::infinity());

  thrust::fill(thrust::cuda::par.on(stream_),
               d_query_write_counters.data_handle(),
               d_query_write_counters.data_handle() + num_queries,
               0);

  if (use_block_sort) {
    thrust::fill(thrust::cuda::par.on(stream_),
                 d_topk_threshold_batch.data_handle(),
                 d_topk_threshold_batch.data_handle() + num_queries,
                 std::numeric_limits<float>::infinity());
  }

  // Launch modified kernel with packed queries instead of LUT
  size_t num_pairs = num_queries * nprobe;
  uint32_t gridDim{static_cast<uint32_t>(num_pairs)};
  uint32_t blockDim{256};

  // Recalculate shared memory for new approach
  size_t query_storage = D * sizeof(float);  // For shared query vector
  const int queue_buffer_smem_bytes =
    use_block_sort ? raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(
                       blockDim / raft::WarpSize, kMaxTopKBlockSort)
                   : 0;

  // Now we need: packed query bits, candidate storage, and query vector
  // this part is also used to store ip2 results
  size_t packed_query_size = max((use_block_sort ? (num_bits * num_words * sizeof(uint32_t)) : 0),
                                 max_cluster_size * sizeof(float));
  size_t candidate_storage = use_block_sort ? max_cluster_size * (sizeof(float) + sizeof(int)) : 0;
  size_t shared_mem_size =
    max(packed_query_size + candidate_storage + query_storage, (size_t)queue_buffer_smem_bytes);

  ComputeInnerProductsKernelParams kernelParams;
  kernelParams.d_sorted_pairs          = d_sorted_pairs;
  kernelParams.d_query                 = d_query;
  kernelParams.d_short_data            = cur_ivf.get_short_data_device();
  kernelParams.d_cluster_meta          = d_cluster_meta;
  kernelParams.d_packed_queries        = d_packed_queries.data_handle();
  kernelParams.d_widths                = d_widths.data_handle();
  kernelParams.d_short_factors         = cur_ivf.get_short_factors_batch_device();
  kernelParams.d_G_k1xSumq             = d_G_k1xSumq;
  kernelParams.d_G_kbxSumq             = d_G_kbxSumq;
  kernelParams.d_centroid_distances    = get_centroid_distances();
  kernelParams.topk                    = topk;
  kernelParams.num_queries             = num_queries;
  kernelParams.nprobe                  = nprobe;
  kernelParams.num_pairs               = num_pairs;
  kernelParams.num_centroids           = cur_ivf.get_num_centroids();
  kernelParams.D                       = D;
  kernelParams.d_threshold             = d_topk_threshold_batch.data_handle();
  kernelParams.max_candidates_per_pair = max_cluster_size;
  kernelParams.max_candidates_per_query =
    use_block_sort ? 0 /* unused */ : max_probed_vectors_count.value();
  kernelParams.ex_bits      = cur_ivf.get_ex_bits();
  kernelParams.d_long_code  = cur_ivf.get_long_code_device();
  kernelParams.d_ex_factor  = reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device());
  kernelParams.d_pids       = cur_ivf.get_ids_device();
  kernelParams.d_topk_dists = d_topk_dists.data_handle();
  kernelParams.d_topk_pids  = d_topk_pids.data_handle();
  kernelParams.d_query_write_counters = d_query_write_counters.data_handle();
  kernelParams.num_bits               = num_bits;
  kernelParams.num_words              = num_words;

  if (!use_4bit) {
    if (cur_ivf.get_ex_bits() != 0) {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseBlockSort<8, true>
                                   : computeInnerProductsWithBitwiseOpt;
      auto const& kernel_launcher = [&](auto const& kernel) -> void {
        kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
      };
      cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
        kernel, shared_mem_size, kernel_launcher);
    } else {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseBlockSort<8, false>
                                   : computeInnerProductsWithBitwiseOptNoEX;
      auto const& kernel_launcher = [&](auto const& kernel) -> void {
        kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
      };
      cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
        kernel, shared_mem_size, kernel_launcher);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {
    if (cur_ivf.get_ex_bits() != 0) {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseBlockSort<4, true>
                                   : computeInnerProductsWithBitwiseOpt;
      auto const& kernel_launcher = [&](auto const& kernel) -> void {
        kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
      };
      cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
        kernel, shared_mem_size, kernel_launcher);
    } else {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseBlockSort<4, false>
                                   : computeInnerProductsWithBitwiseOptNoEX;
      auto const& kernel_launcher = [&](auto const& kernel) -> void {
        kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
      };
      cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
        kernel, shared_mem_size, kernel_launcher);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // Merge results
  raft::matrix::detail::select_k(
    handle_,
    d_topk_dists.data_handle(),
    d_topk_pids.data_handle(),
    num_queries,
    use_block_sort ? (nprobe * topk) : max_probed_vectors_count.value(),
    topk,
    d_final_dists,
    d_final_pids,
    /*select_min = */ true,
    /* sorted = */ false);

  raft::resource::sync_stream(handle_);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
