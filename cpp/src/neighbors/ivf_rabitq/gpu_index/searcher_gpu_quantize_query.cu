/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

// This file implements `SearcherGPU::SearchClusterQueryPairsQuantizeQuery`.
#include "../../ivf_flat/ivf_flat_interleaved_scan.cuh"
#include "searcher_gpu.cuh"
#include "searcher_gpu_common.cuh"

#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select_k.cuh>

#include <thrust/fill.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>

namespace cuvs::neighbors::ivf_rabitq::detail {

__global__ void computeInnerProductsWithBitwiseOpt8bit(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx < num_vectors_in_cluster) {
      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];

        // Fully unrolled bit processing for better ILP
        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator += __popc(shared_packed_query[3 * params.num_words + word] & data_word) << 3;
        accumulator += __popc(shared_packed_query[4 * params.num_words + word] & data_word) << 4;
        accumulator += __popc(shared_packed_query[5 * params.num_words + word] & data_word) << 5;
        accumulator += __popc(shared_packed_query[6 * params.num_words + word] & data_word) << 6;
        accumulator -= __popc(shared_packed_query[7 * params.num_words + word] & data_word) << 7;
      }

      // Restore scale and compute estimated distance
      float ip = (float)accumulator * query_width;

      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = ip;
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

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx = shared_candidate_indices[cand_idx];

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

        // Store the exact inner product
        shared_candidate_ips[cand_idx] = exact_ip;
      }
    }

    __syncthreads();
    //    ------------------

    __shared__ int probe_slot;
    uint32_t output_offset;
    {
      // Additional shared values needed for Step 3
      __shared__ float q_kbxsumq;
      if (tid == 0) { q_kbxsumq = params.d_G_kbxSumq[query_idx]; }
      __syncthreads();

      // Calculate long code parameters
      const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

      // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

      // Reuse shared_candidate_dists to store IP2 results
      float* shared_ip2_results = reinterpret_cast<float*>(shared_mem_raw_2);

      const int warp_id   = tid / WARP_SIZE;
      const int lane_id   = tid % WARP_SIZE;
      const int num_warps = num_threads / WARP_SIZE;

      // Each warp processes different candidates
      for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
        size_t global_vec_idx = cluster_start_index + shared_candidate_indices[cand_idx];

        // Pointer to this vector's long code
        const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

        // Warp-level IP2 computation
        float ip2 = 0.0f;

        // Each thread in warp processes different dimensions
        for (uint32_t d = lane_id; d < params.D; d += WARP_SIZE) {
          // Extract ex_bits value for this dimension
          uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits);
          float ex_val      = (float)code_val;
          ip2 += shared_query[d] * ex_val;
        }

        // Warp-level reduction for ip2
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
        }

        // Lane 0 stores the result
        if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
      }

      __syncthreads();

      // Step 3 Part 2: Each thread computes final distance and writes to output

      // Calculate how many rounds we need
      const int adds_per_thread = (num_candidates + num_threads - 1) / num_threads;
      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
      __syncthreads();
      // Calculate output offset
      output_offset = query_idx * (params.max_candidates_per_pair * params.nprobe) +
                      probe_slot * params.max_candidates_per_pair;

      for (int round = 0; round < adds_per_thread; round++) {
        int cand_idx = tid + round * num_threads;

        float ex_dist;

        if (cand_idx < num_candidates) {
          // Get pre-computed values
          float ip              = shared_candidate_ips[cand_idx];
          float ip2             = shared_ip2_results[cand_idx];
          int local_vec_idx     = shared_candidate_indices[cand_idx];
          size_t global_vec_idx = cluster_start_index + local_vec_idx;

          // vec load version
          float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);

          // Write to global memory
          params.d_topk_dists[output_offset + cand_idx] = ex_dist;
          params.d_topk_pids[output_offset + cand_idx]  = (uint32_t)params.d_pids[global_vec_idx];
        }
      }
      __syncthreads();
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt8bitBlockSort(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;
  __shared__ float q_g_error;
  __shared__ float threshold;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    q_g_error      = sqrtf(q_g_add);
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    threshold      = params.d_threshold[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate        = false;
    float local_ip_quantized = 0;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];

        // Fully unrolled bit processing for better ILP
        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator += __popc(shared_packed_query[3 * params.num_words + word] & data_word) << 3;
        accumulator += __popc(shared_packed_query[4 * params.num_words + word] & data_word) << 4;
        accumulator += __popc(shared_packed_query[5 * params.num_words + word] & data_word) << 5;
        accumulator += __popc(shared_packed_query[6 * params.num_words + word] & data_word) << 6;
        accumulator -= __popc(shared_packed_query[7 * params.num_words + word] & data_word) << 7;
      }

      // Restore scale and compute estimated distance
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

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx = shared_candidate_indices[cand_idx];

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

        // Store the exact inner product
        shared_candidate_ips[cand_idx] = exact_ip;
      }
    }

    __syncthreads();
    //    ------------------

    __shared__ int probe_slot;
    {
      using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
        flat_block_sort<MAX_TOP_K_BLOCK_SORT, true, T, IdxT>::type;
      block_sort_t queue(params.topk);

      // Additional shared values needed for Step 3
      __shared__ float q_kbxsumq;
      if (tid == 0) { q_kbxsumq = params.d_G_kbxSumq[query_idx]; }
      __syncthreads();

      // Calculate long code parameters
      const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

      // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

      // Reuse shared_candidate_dists to store IP2 results
      float* shared_ip2_results = reinterpret_cast<float*>(shared_mem_raw_2);

      const int warp_id   = tid / WARP_SIZE;
      const int lane_id   = tid % WARP_SIZE;
      const int num_warps = num_threads / WARP_SIZE;

      // Each warp processes different candidates
      for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
        size_t global_vec_idx = cluster_start_index + shared_candidate_indices[cand_idx];

        // Pointer to this vector's long code
        const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

        // Warp-level IP2 computation
        float ip2 = 0.0f;

        // Each thread in warp processes different dimensions
        for (uint32_t d = lane_id; d < params.D; d += WARP_SIZE) {
          // Extract ex_bits value for this dimension
          uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits);
          float ex_val      = (float)code_val;
          ip2 += shared_query[d] * ex_val;
        }

        // Warp-level reduction for ip2
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
        }

        // Lane 0 stores the result
        if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
      }

      __syncthreads();

      // Step 3 Part 2: Each thread computes final distance and adds to queue
      // Step 3 Part 2: FIXED - Ensure all threads call queue.add() the same number of times

      // Calculate how many rounds we need (all threads must do the same number of adds)
      const int adds_per_thread = (num_candidates + num_threads - 1) / num_threads;

      for (int round = 0; round < adds_per_thread; round++) {
        int cand_idx = tid + round * num_threads;

        float ex_dist;
        uint32_t pid;

        if (cand_idx < num_candidates) {
          // Get pre-computed values
          float ip              = shared_candidate_ips[cand_idx];
          float ip2             = shared_ip2_results[cand_idx];
          int local_vec_idx     = shared_candidate_indices[cand_idx];
          size_t global_vec_idx = cluster_start_index + local_vec_idx;

          // vec load version
          float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);
          //
          // Get PID
          pid = (uint32_t)params.d_pids[global_vec_idx];

        } else {
          // Thread has no valid candidate for this round - use dummy values
          ex_dist = INFINITY;
          pid     = 0;
        }
        // ALL threads call queue.add() exactly once per round
        queue.add(ex_dist, pid);
      }

      __syncthreads();

      // Step 3 Part 3: Merge results and write back top-k
      queue.done((uint8_t*)shared_mem_raw_2);

      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= params.nprobe) { return; }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
      queue.store(params.d_topk_dists + output_offset,
                  (uint32_t*)(params.d_topk_pids + output_offset));
    }

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
    if (num_candidates >= params.topk) {
      float max_topk_dist;

      if (tid == 0) {
        max_topk_dist = -INFINITY;

        // Find the maximum distance in our top-k results
        uint32_t output_offset =
          query_idx * (params.topk * params.nprobe) +
          probe_slot * params.topk;  // <-- Use probe_slot, not (block_id % nprobe)

        for (uint32_t i = 0; i < params.topk; i++) {
          float dist = params.d_topk_dists[output_offset + i];
          if (dist > 0 && dist > max_topk_dist && dist < INFINITY) { max_topk_dist = dist; }
        }
      }

      __syncthreads();

      // Update threshold using atomicMin (for floats)
      // max_topk_dist should be > 0 to prevent using initialized memory
      if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
        // Use integer interpretation for atomic operations
        int* threshold_ptr = (int*)(params.d_threshold + query_idx);
        int new_val        = __float_as_int(max_topk_dist);

        // Atomic minimum for floats (assuming positive distances)
        atomicMin(threshold_ptr, new_val);

        // Note: atomicMin on int representation works correctly for positive floats
        // because IEEE 754 float format preserves ordering for positive values
      }
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt8bitNoEX(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Ori version --------------------------------------
  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx < num_vectors_in_cluster) {
      // size_t factor_offset = cluster_start_index + vec_idx;
      // float3 factors       = reinterpret_cast<const
      // float3*>(params.d_short_factors)[factor_offset]; float f_add          = factors.x; float
      // f_rescale      = factors.y; float f_error        = factors.z;

      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];

        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator += __popc(shared_packed_query[3 * params.num_words + word] & data_word) << 3;
        accumulator += __popc(shared_packed_query[4 * params.num_words + word] & data_word) << 4;
        accumulator += __popc(shared_packed_query[5 * params.num_words + word] & data_word) << 5;
        accumulator += __popc(shared_packed_query[6 * params.num_words + word] & data_word) << 6;
        accumulator -= __popc(shared_packed_query[7 * params.num_words + word] & data_word)
                       << 7;  // Sign bit
      }

      // Restore scale and compute estimated distance
      float ip = (float)accumulator * query_width;

      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  // -----------------

  __syncthreads();

  if (num_candidates > 0) {
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;
    // Atomically get write position
    __shared__ int probe_slot;
    if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
    __syncthreads();
    // Calculate output offset
    size_t output_offset = query_idx * (params.max_candidates_per_pair * params.nprobe) +
                           probe_slot * params.max_candidates_per_pair;

    float final_1bit_dist;
    PID final_1bit_pid;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx          = shared_candidate_indices[cand_idx];
        size_t factor_offset = cluster_start_index + vec_idx;
        float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
        float f_add     = factors.x;
        float f_rescale = factors.y;
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
        final_1bit_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
        final_1bit_pid  = (uint32_t)params.d_pids[global_vec_idx];

        // Write to global memory
        params.d_topk_dists[output_offset + cand_idx] = final_1bit_dist;
        params.d_topk_pids[output_offset + cand_idx]  = final_1bit_pid;
      }
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt8bitNoEXBlockSort(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;
  __shared__ float q_g_error;
  __shared__ float threshold;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    q_g_error      = sqrtf(q_g_add);
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    threshold      = params.d_threshold[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Ori version --------------------------------------
  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate        = false;
    float local_ip_quantized = 0;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];

        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator += __popc(shared_packed_query[3 * params.num_words + word] & data_word) << 3;
        accumulator += __popc(shared_packed_query[4 * params.num_words + word] & data_word) << 4;
        accumulator += __popc(shared_packed_query[5 * params.num_words + word] & data_word) << 5;
        accumulator += __popc(shared_packed_query[6 * params.num_words + word] & data_word) << 6;
        accumulator -= __popc(shared_packed_query[7 * params.num_words + word] & data_word)
                       << 7;  // Sign bit
      }

      // Restore scale and compute estimated distance
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
  // -----------------

  __syncthreads();

  if (num_candidates > 0) {
    using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K_BLOCK_SORT, true, T, IdxT>::type;
    block_sort_t queue(params.topk);

    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;
    float final_1bit_dist;
    PID final_1bit_pid;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx          = shared_candidate_indices[cand_idx];
        size_t factor_offset = cluster_start_index + vec_idx;
        float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
        float f_add     = factors.x;
        float f_rescale = factors.y;
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

        // get final results and push to queue
        final_1bit_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
        final_1bit_pid  = (uint32_t)params.d_pids[global_vec_idx];

      } else {
        final_1bit_dist = INFINITY;
        final_1bit_pid  = 0;
      };
      queue.add(final_1bit_dist, final_1bit_pid);
    }

    __syncthreads();
    //    ------------------
    __shared__ int probe_slot;
    queue.done((uint8_t*)shared_mem_raw_2);

    // Atomically get write position
    if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
    __syncthreads();

    // Calculate output offset and store results
    uint32_t output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
    queue.store(params.d_topk_dists + output_offset,
                (uint32_t*)(params.d_topk_pids + output_offset));

    // Step 3: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
    if (num_candidates >= params.topk) {
      float max_topk_dist;

      if (tid == 0) {
        max_topk_dist = -INFINITY;

        // Find the maximum distance in our top-k results
        uint32_t output_offset =
          query_idx * (params.topk * params.nprobe) +
          probe_slot * params.topk;  // <-- Use probe_slot, not (block_id % nprobe)

        for (uint32_t i = 0; i < params.topk; i++) {
          float dist = params.d_topk_dists[output_offset + i];
          if (dist > 0 && dist > max_topk_dist && dist < INFINITY) { max_topk_dist = dist; }
        }
      }

      __syncthreads();

      // Update threshold using atomicMin (for floats)
      // max_topk_dist should be > 0 to prevent using initialized memory
      if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
        // Use integer interpretation for atomic operations
        int* threshold_ptr = (int*)(params.d_threshold + query_idx);
        int new_val        = __float_as_int(max_topk_dist);

        // Atomic minimum for floats (assuming positive distances)
        atomicMin(threshold_ptr, new_val);

        // Note: atomicMin on int representation works correctly for positive floats
        // because IEEE 754 float format preserves ordering for positive values
      }
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt4bit(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Ori version --------------------------------------
  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx < num_vectors_in_cluster) {
      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      int32_t accumulator2 = 0;
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];
        accumulator2 += __popc(data_word);

        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator -= __popc(shared_packed_query[3 * params.num_words + word] & data_word)
                       << 3;  // Sign bit
      }

      // Restore scale and compute estimated distance
      float ip = (float)accumulator * query_width;

      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  // -----------------

  __syncthreads();

  if (num_candidates > 0) {
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx = shared_candidate_indices[cand_idx];

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

        // Store the exact inner product
        shared_candidate_ips[cand_idx] = exact_ip;
      }
    }

    __syncthreads();
    //    ------------------

    __shared__ int probe_slot;
    uint32_t output_offset;
    {
      // Additional shared values needed for Step 3
      __shared__ float q_kbxsumq;
      if (tid == 0) { q_kbxsumq = params.d_G_kbxSumq[query_idx]; }
      __syncthreads();

      // Calculate long code parameters
      const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

      // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

      // Reuse shared_candidate_dists to store IP2 results
      float* shared_ip2_results = reinterpret_cast<float*>(shared_mem_raw_2);

      const int warp_id   = tid / WARP_SIZE;
      const int lane_id   = tid % WARP_SIZE;
      const int num_warps = num_threads / WARP_SIZE;

      // Each warp processes different candidates
      for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
        size_t global_vec_idx = cluster_start_index + shared_candidate_indices[cand_idx];

        // Pointer to this vector's long code
        const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

        // Warp-level IP2 computation
        float ip2 = 0.0f;

        // Each thread in warp processes different dimensions
        for (uint32_t d = lane_id; d < params.D; d += WARP_SIZE) {
          // Extract ex_bits value for this dimension
          uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits);
          float ex_val      = (float)code_val;
          ip2 += shared_query[d] * ex_val;
        }

        // Warp-level reduction for ip2
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
        }

        // Lane 0 stores the result
        if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
      }

      __syncthreads();

      // Step 3 Part 2: Each thread computes final distance and writes to output

      // Calculate how many rounds we need (all threads must do the same number of adds)
      const int adds_per_thread = (num_candidates + num_threads - 1) / num_threads;
      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
      __syncthreads();
      // Calculate output offset
      output_offset = query_idx * (params.max_candidates_per_pair * params.nprobe) +
                      probe_slot * params.max_candidates_per_pair;

      for (int round = 0; round < adds_per_thread; round++) {
        int cand_idx = tid + round * num_threads;

        float ex_dist;

        if (cand_idx < num_candidates) {
          // Get pre-computed values
          float ip              = shared_candidate_ips[cand_idx];
          float ip2             = shared_ip2_results[cand_idx];
          int local_vec_idx     = shared_candidate_indices[cand_idx];
          size_t global_vec_idx = cluster_start_index + local_vec_idx;

          // vec load version
          float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);

          // Write to global memory
          params.d_topk_dists[output_offset + cand_idx] = ex_dist;
          params.d_topk_pids[output_offset + cand_idx]  = (uint32_t)params.d_pids[global_vec_idx];
        }
      }
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt4bitBlockSort(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;
  __shared__ float q_g_error;
  __shared__ float threshold;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    q_g_error      = sqrtf(q_g_add);
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    threshold      = params.d_threshold[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Ori version --------------------------------------
  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate        = false;
    float local_ip_quantized = 0;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      int32_t accumulator2 = 0;
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];
        accumulator2 += __popc(data_word);

        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator -= __popc(shared_packed_query[3 * params.num_words + word] & data_word)
                       << 3;  // Sign bit
      }

      // Restore scale and compute estimated distance
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
  // -----------------

  __syncthreads();

  if (num_candidates > 0) {
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx = shared_candidate_indices[cand_idx];

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

        // Store the exact inner product
        shared_candidate_ips[cand_idx] = exact_ip;
      }
    }

    __syncthreads();
    //    ------------------

    __shared__ int probe_slot;
    {
      using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
        flat_block_sort<MAX_TOP_K_BLOCK_SORT, true, T, IdxT>::type;
      block_sort_t queue(params.topk);

      // Additional shared values needed for Step 3
      __shared__ float q_kbxsumq;
      if (tid == 0) { q_kbxsumq = params.d_G_kbxSumq[query_idx]; }
      __syncthreads();

      // Calculate long code parameters
      const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

      // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

      // Reuse shared_candidate_dists to store IP2 results
      float* shared_ip2_results = reinterpret_cast<float*>(shared_mem_raw_2);

      const int warp_id   = tid / WARP_SIZE;
      const int lane_id   = tid % WARP_SIZE;
      const int num_warps = num_threads / WARP_SIZE;

      // Each warp processes different candidates
      for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
        size_t global_vec_idx = cluster_start_index + shared_candidate_indices[cand_idx];

        // Pointer to this vector's long code
        const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

        // Warp-level IP2 computation
        float ip2 = 0.0f;

        // Each thread in warp processes different dimensions
        for (uint32_t d = lane_id; d < params.D; d += WARP_SIZE) {
          // Extract ex_bits value for this dimension
          uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits);
          float ex_val      = (float)code_val;
          ip2 += shared_query[d] * ex_val;
        }

        // Warp-level reduction for ip2
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
        }

        // Lane 0 stores the result
        if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
      }

      __syncthreads();

      // Step 3 Part 2: Each thread computes final distance and adds to queue
      // Step 3 Part 2: FIXED - Ensure all threads call queue.add() the same number of times

      // Calculate how many rounds we need (all threads must do the same number of adds)
      const int adds_per_thread = (num_candidates + num_threads - 1) / num_threads;

      for (int round = 0; round < adds_per_thread; round++) {
        int cand_idx = tid + round * num_threads;

        float ex_dist;
        uint32_t pid;

        if (cand_idx < num_candidates) {
          // Get pre-computed values
          float ip              = shared_candidate_ips[cand_idx];
          float ip2             = shared_ip2_results[cand_idx];
          int local_vec_idx     = shared_candidate_indices[cand_idx];
          size_t global_vec_idx = cluster_start_index + local_vec_idx;

          // vec load version
          float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);

          // Get PID
          pid = (uint32_t)params.d_pids[global_vec_idx];

        } else {
          // Thread has no valid candidate for this round - use dummy values
          ex_dist = INFINITY;
          pid     = 0;
        }
        // ALL threads call queue.add() exactly once per round
        queue.add(ex_dist, pid);
      }

      __syncthreads();

      // Step 3 Part 3: Merge results and write back top-k
      queue.done((uint8_t*)shared_mem_raw_2);

      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= params.nprobe) { return; }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
      queue.store(params.d_topk_dists + output_offset,
                  (uint32_t*)(params.d_topk_pids + output_offset));
    }

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
    if (num_candidates >= params.topk) {
      float max_topk_dist;

      if (tid == 0) {
        max_topk_dist = -INFINITY;

        // Find the maximum distance in our top-k results
        uint32_t output_offset =
          query_idx * (params.topk * params.nprobe) +
          probe_slot * params.topk;  // <-- Use probe_slot, not (block_id % nprobe)

        for (uint32_t i = 0; i < params.topk; i++) {
          float dist = params.d_topk_dists[output_offset + i];
          if (dist > 0 && dist > max_topk_dist && dist < INFINITY) { max_topk_dist = dist; }
        }
      }

      __syncthreads();

      // Update threshold using atomicMin (for floats)
      // max_topk_dist should be > 0 to prevent using initialized memory
      if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
        // Use integer interpretation for atomic operations
        int* threshold_ptr = (int*)(params.d_threshold + query_idx);
        int new_val        = __float_as_int(max_topk_dist);

        // Atomic minimum for floats (assuming positive distances)
        atomicMin(threshold_ptr, new_val);

        // Note: atomicMin on int representation works correctly for positive floats
        // because IEEE 754 float format preserves ordering for positive values
      }
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt4bitNoEX(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Ori version --------------------------------------
  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx < num_vectors_in_cluster) {
      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];

        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator -= __popc(shared_packed_query[3 * params.num_words + word] & data_word)
                       << 3;  // Sign bit
      }

      // Restore scale and compute estimated distance
      float ip = (float)accumulator * query_width;

      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  // -----------------

  __syncthreads();

  if (num_candidates > 0) {
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;
    // Atomically get write position
    __shared__ int probe_slot;
    if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
    __syncthreads();
    // Calculate output offset
    uint32_t output_offset = query_idx * (params.max_candidates_per_pair * params.nprobe) +
                             probe_slot * params.max_candidates_per_pair;

    float final_1bit_dist;
    PID final_1bit_pid;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx          = shared_candidate_indices[cand_idx];
        size_t factor_offset = cluster_start_index + vec_idx;
        float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
        float f_add     = factors.x;
        float f_rescale = factors.y;
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
        final_1bit_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
        final_1bit_pid  = (uint32_t)params.d_pids[global_vec_idx];
        // Write to global memory
        params.d_topk_dists[output_offset + cand_idx] = final_1bit_dist;
        params.d_topk_pids[output_offset + cand_idx]  = final_1bit_pid;
      }
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt4bitNoEXBlockSort(
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

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr =
    params.d_packed_queries + query_idx * params.num_bits * params.num_words;
  for (uint32_t i = tid; i < params.num_bits * params.num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = params.d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;
  __shared__ float q_g_error;
  __shared__ float threshold;

  if (tid == 0) {
    q_g_add        = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    q_g_error      = sqrtf(q_g_add);
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    threshold      = params.d_threshold[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes   = max(params.num_bits * params.num_words * sizeof(uint32_t),
                                  params.max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
  float* shared_query = (float*)(shared_candidate_indices + params.max_candidates_per_pair);
  const size_t short_code_length = params.D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Ori version --------------------------------------
  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate        = false;
    float local_ip_quantized = 0;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      for (int word = 0; word < params.num_words; ++word) {
        size_t data_offset =
          cluster_start_index * params.num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = params.d_short_data[data_offset];

        accumulator += __popc(shared_packed_query[0 * params.num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * params.num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * params.num_words + word] & data_word) << 2;
        accumulator -= __popc(shared_packed_query[3 * params.num_words + word] & data_word)
                       << 3;  // Sign bit
      }

      // Restore scale and compute estimated distance
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
  // -----------------

  __syncthreads();

  if (num_candidates > 0) {
    using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K_BLOCK_SORT, true, T, IdxT>::type;
    block_sort_t queue(params.topk);

    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;
    float final_1bit_dist;
    PID final_1bit_pid;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
        int vec_idx          = shared_candidate_indices[cand_idx];
        size_t factor_offset = cluster_start_index + vec_idx;
        float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
        float f_add     = factors.x;
        float f_rescale = factors.y;
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

        // get final results and push to queue
        final_1bit_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
        final_1bit_pid  = (uint32_t)params.d_pids[global_vec_idx];

      } else {
        final_1bit_dist = INFINITY;
        final_1bit_pid  = 0;
      };
      queue.add(final_1bit_dist, final_1bit_pid);
    }

    __syncthreads();
    //    ------------------
    __shared__ int probe_slot;
    queue.done((uint8_t*)shared_mem_raw_2);

    // Atomically get write position
    if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
    __syncthreads();

    // Calculate output offset and store results
    uint32_t output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
    queue.store(params.d_topk_dists + output_offset,
                (uint32_t*)(params.d_topk_pids + output_offset));

    // Step 3: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
    if (num_candidates >= params.topk) {
      float max_topk_dist;

      if (tid == 0) {
        max_topk_dist = -INFINITY;

        // Find the maximum distance in our top-k results
        uint32_t output_offset =
          query_idx * (params.topk * params.nprobe) +
          probe_slot * params.topk;  // <-- Use probe_slot, not (block_id % nprobe)

        for (uint32_t i = 0; i < params.topk; i++) {
          float dist = params.d_topk_dists[output_offset + i];
          if (dist > 0 && dist > max_topk_dist && dist < INFINITY) { max_topk_dist = dist; }
        }
      }

      __syncthreads();

      // Update threshold using atomicMin (for floats)
      // max_topk_dist should be > 0 to prevent using initialized memory
      if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
        // Use integer interpretation for atomic operations
        int* threshold_ptr = (int*)(params.d_threshold + query_idx);
        int new_val        = __float_as_int(max_topk_dist);

        // Atomic minimum for floats (assuming positive distances)
        atomicMin(threshold_ptr, new_val);

        // Note: atomicMin on int representation works correctly for positive floats
        // because IEEE 754 float format preserves ordering for positive values
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

  typedef cub::BlockReduce<float, 256> BlockReduceFloat;
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
  float* d_topk_dists,
  PID* d_topk_pids,
  float* d_final_dists,
  PID* d_final_pids,
  bool use_4bit  // Add parameter to choose 4-bit or 8-bit
)
{
  // choose algorithm
  const bool use_block_sort{topk <= MAX_TOP_K_BLOCK_SORT};

  // query quantize
  const int num_bits  = use_4bit ? 4 : 8;  // Choose bit width
  const int num_words = (cur_ivf.get_num_padded_dim() + 31) / 32;

  // Allocate memory for quantization
  size_t ranges_size     = num_queries * 2 * sizeof(float);
  size_t widths_size     = num_queries * sizeof(float);
  size_t quantized_size  = num_queries * cur_ivf.get_num_padded_dim() * sizeof(int8_t);
  size_t packed_size     = num_queries * num_bits * num_words * sizeof(uint32_t);
  size_t counters_size   = num_queries * sizeof(int);
  size_t thresholds_size = use_block_sort ? num_queries * sizeof(float) : 0;

  auto align4 = [](size_t x) { return (x + 3) & ~size_t(3); };

  size_t workspace_size = 0;
  workspace_size += align4(ranges_size);
  workspace_size += align4(widths_size);
  workspace_size += align4(quantized_size);
  workspace_size += align4(packed_size);
  workspace_size += align4(counters_size);
  workspace_size += align4(thresholds_size);

  uint8_t* d_workspace = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_workspace, workspace_size, stream_));

  uint8_t* ptr = d_workspace;

  float* d_query_ranges = reinterpret_cast<float*>(ptr);
  ptr += align4(ranges_size);

  float* d_widths = reinterpret_cast<float*>(ptr);
  ptr += align4(widths_size);

  int8_t* d_quantized_queries = reinterpret_cast<int8_t*>(ptr);
  ptr += align4(quantized_size);

  uint32_t* d_packed_queries = reinterpret_cast<uint32_t*>(ptr);
  ptr += align4(packed_size);

  int* d_query_write_counters = reinterpret_cast<int*>(ptr);
  ptr += align4(counters_size);

  float* d_topk_threshold_batch = reinterpret_cast<float*>(ptr);
  ptr += align4(thresholds_size);

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
                                                       d_quantized_queries,
                                                       d_widths);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {  // scalar quantize
    // Step 1: Find min/max for each query
    {
      dim3 block(256);
      dim3 grid(num_queries);
      findQueryRanges<<<grid, block, 0, stream_>>>(
        d_query, d_query_ranges, num_queries, cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    // Step 2: Quantize queries to int8_t with BQ=8
    {
      const int block_size = 256;
      const int grid_size  = num_queries;
      if (use_4bit) {
        quantizeQueriesToInt4<<<grid_size, block_size, 0, stream_>>>(d_query,
                                                                     d_query_ranges,
                                                                     d_quantized_queries,
                                                                     d_widths,
                                                                     num_queries,
                                                                     cur_ivf.get_num_padded_dim());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      } else {
        quantizeQueriesToInt8<<<grid_size, block_size, 0, stream_>>>(d_query,
                                                                     d_query_ranges,
                                                                     d_quantized_queries,
                                                                     d_widths,
                                                                     num_queries,
                                                                     cur_ivf.get_num_padded_dim());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    }
  }

  // Step 3: Pack quantized queries into bit planes
  {
    const int block_size = 256;
    const int grid_size  = (num_queries * num_bits * num_words + block_size - 1) / block_size;

    if (use_4bit) {
      packInt4QueryBitPlanes<<<grid_size, block_size, 0, stream_>>>(
        d_quantized_queries, d_packed_queries, num_queries, cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {
      packInt8QueryBitPlanes<<<grid_size, block_size, 0, stream_>>>(
        d_quantized_queries, d_packed_queries, num_queries, cur_ivf.get_num_padded_dim());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }

  // Initialize distances
  size_t max_cluster_length                 = cur_ivf.get_max_cluster_length();
  size_t max_candidates_per_query_per_probe = use_block_sort ? topk : max_cluster_length;
  size_t total_elements = max_candidates_per_query_per_probe * num_queries * nprobe;
  thrust::fill(thrust::cuda::par.on(stream_),
               d_topk_dists,
               d_topk_dists + total_elements,
               std::numeric_limits<float>::infinity());

  RAFT_CUDA_TRY(cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream_));

  if (use_block_sort) {
    thrust::fill(thrust::cuda::par.on(stream_),
                 d_topk_threshold_batch,
                 d_topk_threshold_batch + num_queries,
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
                       blockDim / WARP_SIZE, MAX_TOP_K_BLOCK_SORT)
                   : 0;

  // Now we need: packed query bits, candidate storage, and query vector
  // this part is also used to store ip2 results
  size_t packed_query_size =
    max(num_bits * num_words * sizeof(uint32_t), max_cluster_length * sizeof(float));
  size_t candidate_storage = max_cluster_length * (sizeof(float) + sizeof(int));
  size_t shared_mem_size   = max(packed_query_size + candidate_storage + query_storage +
                                 10 * sizeof(float),  // +sizeof(float) for width
                               (size_t)queue_buffer_smem_bytes);

  ComputeInnerProductsKernelParams kernelParams;
  kernelParams.d_sorted_pairs          = d_sorted_pairs;
  kernelParams.d_query                 = d_query;
  kernelParams.d_short_data            = cur_ivf.get_short_data_device();
  kernelParams.d_cluster_meta          = d_cluster_meta;
  kernelParams.d_packed_queries        = d_packed_queries;
  kernelParams.d_widths                = d_widths;
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
  kernelParams.d_threshold             = d_topk_threshold_batch;
  kernelParams.max_candidates_per_pair = max_cluster_length;
  kernelParams.ex_bits                 = cur_ivf.get_ex_bits();
  kernelParams.d_long_code             = cur_ivf.get_long_code_device();
  kernelParams.d_ex_factor  = reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device());
  kernelParams.d_pids       = cur_ivf.get_ids_device();
  kernelParams.d_topk_dists = d_topk_dists;
  kernelParams.d_topk_pids  = d_topk_pids;
  kernelParams.d_query_write_counters = d_query_write_counters;
  kernelParams.num_bits               = num_bits;
  kernelParams.num_words              = num_words;

  if (!use_4bit) {
    if (cur_ivf.get_ex_bits() != 0) {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseOpt8bitBlockSort
                                   : computeInnerProductsWithBitwiseOpt8bit;
      RAFT_CUDA_TRY(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
      kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
    } else {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseOpt8bitNoEXBlockSort
                                   : computeInnerProductsWithBitwiseOpt8bitNoEX;
      RAFT_CUDA_TRY(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
      kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {
    if (cur_ivf.get_ex_bits() != 0) {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseOpt4bitBlockSort
                                   : computeInnerProductsWithBitwiseOpt4bit;
      RAFT_CUDA_TRY(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
      kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
    } else {
      auto kernel = use_block_sort ? computeInnerProductsWithBitwiseOpt4bitNoEXBlockSort
                                   : computeInnerProductsWithBitwiseOpt4bitNoEX;
      RAFT_CUDA_TRY(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
      kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // Merge results
  raft::matrix::detail::select_k(handle_,
                                 d_topk_dists,
                                 d_topk_pids,
                                 num_queries,
                                 nprobe * max_candidates_per_query_per_probe,
                                 topk,
                                 d_final_dists,
                                 d_final_pids,
                                 /*select_min = */ true,
                                 /* sorted = */ false);

  // Cleanup
  RAFT_CUDA_TRY(cudaFreeAsync(d_workspace, stream_););

  raft::resource::sync_stream(handle_);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
