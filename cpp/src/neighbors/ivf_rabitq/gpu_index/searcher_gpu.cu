/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

// This file implements `SearcherGPU::SearchClusterQueryPairs`.
#include "../utils/memory.hpp"
#include "searcher_gpu.cuh"
#include "searcher_gpu_common.cuh"

#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/neighbors/detail/ivf_flat_interleaved_scan.cuh>

#include <thrust/fill.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <string>

namespace cuvs::neighbors::ivf_rabitq::detail {

SearcherGPU::SearcherGPU(raft::resources const& handle,
                         const float* q,
                         size_t d,
                         size_t ex_bits,
                         std::string mode,
                         DataQuantizerGPU::FastQuantizeFactors* fast_quantize_factors,
                         bool rabitq_quantize_flag)
  : D(d), query_(q), rabitq_quantize_flag_(rabitq_quantize_flag), mode_(mode), handle_(handle)
{
  set_unit_q(memory::align_mm<64, float>(D * sizeof(float)));
  set_quant_query(memory::align_mm<64, int16_t>(D * sizeof(int16_t)));
  if (mode_ == "quant4" && fast_quantize_factors != nullptr) {
    best_rescaling_factor =
      fast_quantize_factors->const_scaling_factor_4bit;  // suppose that always quantize query to 4
                                                         // bits (1 + 3) per dim
  } else if (mode_ == "quant8" && fast_quantize_factors != nullptr) {
    best_rescaling_factor = fast_quantize_factors->const_scaling_factor_8bit;
  } else if (!fast_quantize_factors && (mode_ == "quant4" || mode_ == "quant8")) {
    std::cerr << "ERROR: fast_quantize_factors must be set for quant4/quant8 mode" << std::endl;
  }
  raft::resource::sync_stream(handle);
}

void SearcherGPU::AllocateSearcherSpace(const IVFGPU& cur_ivf,
                                        size_t num_queries,
                                        size_t k,
                                        size_t max_nprobes,
                                        size_t max_cluster_length)
{
  centroid_distances_ =
    raft::make_device_vector<float, int64_t>(handle_, num_queries * cur_ivf.get_num_centroids());
  c_norms_ = raft::make_device_vector<float, int64_t>(handle_, cur_ivf.get_num_centroids());
  q_norms_ = raft::make_device_vector<float, int64_t>(handle_, num_queries);
  raft::resource::sync_stream(handle_);
};

__global__ void precomputeAllLUTs(const float* d_query,      // Query vectors
                                  float* d_lut_for_queries,  // Output LUTs for all queries
                                  size_t num_queries,        // Number of queries
                                  size_t D                   // Dimension
)
{
  // Each block handles one query
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Calculate LUT parameters
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Pointer to this query's LUT in global memory
  float* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

  // Pointer to this query's vector
  const float* query_vec = d_query + query_idx * D;

  // Each thread computes part of the LUT
  for (size_t chunk_idx = tid; chunk_idx < num_chunks; chunk_idx += num_threads) {
    size_t dim_start = chunk_idx * BITS_PER_CHUNK;

    // Compute LUT entries for this chunk
    for (int lut_entry = 0; lut_entry < LUT_SIZE; lut_entry++) {
      float sum = 0.0f;

      // For each bit in the 4-bit pattern
      for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
        size_t dim = dim_start + bit_idx;
        if (dim < D) {  // Check if within actual dimension
          // Check if bit is set in the pattern
          if (lut_entry & (1 << (BITS_PER_CHUNK - 1 - bit_idx))) { sum += query_vec[dim]; }
        }
      }

      // Store in global LUT
      size_t lut_offset     = chunk_idx * LUT_SIZE + lut_entry;
      query_lut[lut_offset] = sum;
    }
  }
}

// Launch function for precomputing LUTs
void launchPrecomputeLUTs(const float* d_query,
                          float* d_lut_for_queries,
                          size_t num_queries,
                          size_t D,
                          rmm::cuda_stream_view stream)
{
  // Initialize all LUTs to invalid value first
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;
  const size_t total_lut_size     = num_queries * lut_per_query_size;

  // Optional: Initialize to -infinity to mark as uncomputed
  // (You can skip this if you always call precompute before main kernel)
  float neg_inf = -std::numeric_limits<float>::infinity();
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_lut_for_queries, *reinterpret_cast<int*>(&neg_inf), total_lut_size * sizeof(float), stream));

  // Launch precompute kernel
  dim3 gridDim(num_queries, 1, 1);
  dim3 blockDim(256, 1, 1);  // Can tune this

  precomputeAllLUTs<<<gridDim, blockDim, 0, stream>>>(d_query, d_lut_for_queries, num_queries, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename QueueT, bool SharedQueueBuffer>
__global__ void computeInnerProductsWithLUT(const ComputeInnerProductsKernelParams params)
{
  // Each block handles one <cluster, query> pair
  const int block_id = blockIdx.x;  // simply use 1-D block

  if (block_id >= params.num_pairs) return;

  // Get the cluster-query pair for this block
  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  // Check bounds
  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  // Get cluster metadata
  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  // Calculate LUT parameters
  const size_t num_chunks         = params.D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Shared memory for LUT
  extern __shared__ __align__(256) float shared_lut[];

  // Thread index within the block
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointer to this query's LUT in global memory
  float* query_lut = params.d_lut_for_queries_float + query_idx * lut_per_query_size;

  // ------

  // Step 1 (skipped): Check if LUT needs to be computed and compute if necessary

  // Then Load LUT into shared memory
  // Each thread loads part of the LUT
  for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut[i] = query_lut[i];
  }

  __syncthreads();

  // Step 2 Part 1: Compute distances using LUT && decide candidates

  // Shared values for this <cluster, query> pair
  __shared__ float q_g_add;       // squared distance to centroid
  __shared__ float q_k1xsumq;     // query factor
  __shared__ float q_g_error;     // sqrt(q_g_add)
  __shared__ float threshold;     // threshold for this query
  __shared__ int num_candidates;  // counter for candidates

  // Load shared query-cluster values
  if (tid == 0) {
    // Get squared distance from query to this cluster's centroid
    q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    q_g_error = sqrtf(q_g_add);

    // Get query factor
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    threshold      = params.d_threshold[query_idx];  // NEW: load threshold
    num_candidates = 0;                              // NEW: initialize counter
  }
  __syncthreads();

  // Allocate shared memory for candidate storage (after LUT)
  // Assuming extern shared memory is large enough
  float* shared_candidate_dists = shared_lut + (num_chunks * LUT_SIZE);
  float* shared_candidate_ips   = shared_candidate_dists + params.max_candidates_per_pair;
  int* shared_candidate_indices = (int*)(shared_candidate_ips + params.max_candidates_per_pair);
  int* shared_buffer            = nullptr;
  if constexpr (SharedQueueBuffer)
    shared_buffer = shared_candidate_indices + params.max_candidates_per_pair;

  // Calculate short code parameters
  const size_t short_code_length = params.D / 32;        // number of uint32_t per vector
  const size_t chunks_per_uint32 = 32 / BITS_PER_CHUNK;  // 8 chunks per uint32_t

  // Each thread processes one or more vectors
  // We'll use a grid-stride loop to handle all vectors in the cluster
  const int vectors_per_iteration = num_threads;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    float local_low_dist = INFINITY;
    float local_ip       = 0.0f;
    bool is_candidate    = false;

    if (vec_idx < num_vectors_in_cluster) {
      // Load short factors for this vector
      // vec load for short factors
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      // Compute inner product using LUT
      float ip = 0.0f;

      // Process each uint32_t of the short code
      for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
        // Access short code in transposed layout
        // For transposed layout: vec1[dim0-31], vec2[dim0-31], ..., vecn[dim0-31]
        size_t short_code_offset =
          cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
        uint32_t short_code_chunk = params.d_short_data[short_code_offset];

        // Process 8 4-bit chunks from this uint32_t
        for (int chunk_in_uint32 = 0; chunk_in_uint32 < chunks_per_uint32; chunk_in_uint32++) {
          // Extract 4-bit pattern
          // Note: in uint32_t, lowest dim is at bit 31 (MSB)
          // So we extract from high bits to low bits
          int shift   = 28 - (chunk_in_uint32 * BITS_PER_CHUNK);  // 28, 24, 20, 16, 12, 8, 4, 0
          int pattern = (short_code_chunk >> shift) & 0xF;        // Extract 4 bits

          // Look up in LUT
          size_t lut_chunk_idx = uint32_idx * chunks_per_uint32 + chunk_in_uint32;
          size_t lut_offset    = lut_chunk_idx * LUT_SIZE + pattern;

          // Accumulate inner product
          ip += shared_lut[lut_offset];
        }
      }

      // Compute estimated distance
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

      // Compute lower bound
      float low_dist = est_dist - f_error * q_g_error;

      // Check threshold
      if (low_dist < threshold) {
        is_candidate   = true;
        local_low_dist = est_dist;
        local_ip       = ip;
      }
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < params.max_candidates_per_pair) {
        shared_candidate_dists[candidate_slot]   = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

  // Step 2 Part 2: Determine which candidates to use
  int final_num_candidates = min(num_candidates, (int)params.max_candidates_per_pair);

  // Step 3 opt: Compute more accurate distances and select top-k
  // Opt: warp-level dist and then thread-level ex dist restore

  __syncthreads();
  //    __shared__ int probe_slot;
  if (final_num_candidates > 0) {
    QueueT queue(params.topk);

    // Additional shared values needed for Step 3
    __shared__ float q_kbxsumq;
    if (tid == 0) { q_kbxsumq = params.d_G_kbxSumq[query_idx]; }
    __syncthreads();

    // Calculate long code parameters
    const size_t long_code_size = (params.D * params.ex_bits + 7) / 8;

    // Load query vector to shared memory
    float* shared_query = (float*)(shared_lut);
    for (size_t i = tid; i < params.D; i += num_threads) {
      shared_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

    // Reuse shared_candidate_dists to store IP2 results
    float* shared_ip2_results = shared_candidate_dists;

    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;
    const int num_warps = num_threads / WARP_SIZE;

    // Each warp processes different candidates
    for (int cand_idx = warp_id; cand_idx < final_num_candidates; cand_idx += num_warps) {
      int local_vec_idx     = shared_candidate_indices[cand_idx];
      size_t global_vec_idx = cluster_start_index + local_vec_idx;

      // Pointer to this vector's long code
      const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;

      // Warp-level IP2 computation
      float ip2 = 0.0f;

      // Each thread in warp processes different dimensions
      for (size_t d = lane_id; d < params.D; d += WARP_SIZE) {
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
    const int adds_per_thread = (final_num_candidates + num_threads - 1) / num_threads;

    for (int round = 0; round < adds_per_thread; round++) {
      int cand_idx = tid + round * num_threads;

      float ex_dist;
      uint32_t pid;

      if (cand_idx < final_num_candidates) {
        // Get pre-computed values
        float ip              = shared_candidate_ips[cand_idx];
        float ip2             = shared_ip2_results[cand_idx];
        int local_vec_idx     = shared_candidate_indices[cand_idx];
        size_t global_vec_idx = cluster_start_index + local_vec_idx;

        // Load ex factors for this vector
        // vec load version
        float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
        float f_ex_add     = ex_factors.x;
        float f_ex_rescale = ex_factors.y;

        // Compute final distance using pre-computed ip2
        ex_dist = f_ex_add + q_g_add +
                  f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);
        //                ex_dist = ex_dist+1;
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

    if constexpr (SharedQueueBuffer) {
      queue.done((uint8_t*)shared_buffer);
    } else {
      queue.done();
    }

    // Atomically get write position
    __shared__ int probe_slot;
    if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
    __syncthreads();

    if (probe_slot >= params.nprobe) { return; }

    // Calculate output offset and store results
    size_t output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
    queue.store(params.d_topk_dists + output_offset,
                (uint32_t*)(params.d_topk_pids + output_offset));

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
    __shared__ float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      size_t output_offset =
        query_idx * (params.topk * params.nprobe) +
        probe_slot * params.topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (size_t i = 0; i < params.topk; i++) {
        float dist = params.d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
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

// instantiate the kernel with a queue type that satisfies the capacity requirement
template <int Capacity = MAX_TOP_K_WARP_SORT>
auto instantiateComputeInnerProductsWithLUT(const uint32_t topk)
{
  static_assert(Capacity <= MAX_TOP_K_WARP_SORT,
                "Capacity exceeds maximum supported: " STR(MAX_TOP_K_WARP_SORT));
  if constexpr (Capacity <=
                MAX_TOP_K_BLOCK_SORT) {  // base case: use block_sort (requires shared mem)
    using block_sort_t = typename raft::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K_BLOCK_SORT, true, T, IdxT>::type;
    return computeInnerProductsWithLUT<block_sort_t, /* SharedQueueBuffer = */ true>;
  } else {
    if (topk * WARP_SORT_CAPACITY_FACTOR >
        Capacity) {  // base case: use warp_sort (does not require shared mem)
      using warp_sort_t =
        raft::matrix::detail::select::warpsort::warp_sort_filtered<Capacity, true, T, IdxT>;
      return computeInnerProductsWithLUT<warp_sort_t, /* SharedQueueBuffer = */ false>;
    } else {  // recursive case
      return instantiateComputeInnerProductsWithLUT<Capacity / WARP_SORT_CAPACITY_FACTOR>(topk);
    }
  }
}

template <typename QueueT, bool SharedQueueBuffer>
__global__ void computeInnerProductsWithLUTNoEX(const ComputeInnerProductsKernelParams params)
{
  // Each block handles one <cluster, query> pair
  const int block_id = blockIdx.x;  // simply use 1-D block

  if (block_id >= params.num_pairs) return;

  // Get the cluster-query pair for this block
  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  // Check bounds
  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  // Get cluster metadata
  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  // Calculate LUT parameters
  const uint32_t num_chunks         = params.D / BITS_PER_CHUNK;
  const uint32_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Shared memory for LUT
  extern __shared__ __align__(256) char shared_mem_raw[];
  float* shared_lut = reinterpret_cast<float*>(shared_mem_raw);
  // Calculate offset for other shared arrays after BF16 LUT
  uint32_t lut_bytes = num_chunks * LUT_SIZE * sizeof(float);

  // Thread index within the block
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointer to this query's LUT in global memory
  float* query_lut = params.d_lut_for_queries_float + query_idx * lut_per_query_size;

  // Then Load LUT into shared memory
  // Direct copy of BF16 values
  for (uint32_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut[i] = query_lut[i];
  }

  __syncthreads();

  // Step 2 Part 1: Compute distances using LUT && decide candidates

  // Shared values for this <cluster, query> pair

  __shared__ int num_candidates;  // counter for candidates
  __shared__ float q_g_add;       // squared distance to centroid
  __shared__ float q_k1xsumq;     // query factor
  __shared__ float threshold;     // threshold for this query
  // Load shared query-cluster values
  if (tid == 0) {
    // Get squared distance from query to this cluster's centroid
    q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];

    // Get query factor
    q_k1xsumq      = params.d_G_k1xSumq[query_idx];
    threshold      = params.d_threshold[query_idx];  // NEW: load threshold
    num_candidates = 0;                              // NEW: initialize counter
  }
  __syncthreads();

  // Allocate shared memory for candidate storage (after LUT)
  // Assuming extern shared memory is large enough
  float* shared_candidate_ips   = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
  int* shared_candidate_indices = (int*)(shared_candidate_ips + params.max_candidates_per_pair);

  // Calculate short code parameters
  const uint32_t short_code_length = params.D / 32;        // number of uint32_t per vector
  const uint32_t chunks_per_uint32 = 32 / BITS_PER_CHUNK;  // 8 chunks per uint32_t

  // Each thread processes one or more vectors
  // We'll use a grid-stride loop to handle all vectors in the cluster
  const int vectors_per_iteration = num_threads;

  float final_1bit_dist;
  PID final_1bit_pid;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate = false;

    if (vec_idx < num_vectors_in_cluster) {
      // vec load for short factors
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;

      // Compute inner product using LUT
      float ip = 0.0f;

      // Process each uint32_t of the short code
      for (uint32_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
        // Access short code in transposed layout
        // For transposed layout: vec1[dim0-31], vec2[dim0-31], ..., vecn[dim0-31]
        size_t short_code_offset =
          cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
        uint32_t short_code_chunk = params.d_short_data[short_code_offset];

        // Process 8 4-bit chunks from this uint32_t
        for (int chunk_in_uint32 = 0; chunk_in_uint32 < chunks_per_uint32; chunk_in_uint32++) {
          // Extract 4-bit pattern
          // Note: in uint32_t, lowest dim is at bit 31 (MSB)
          // So we extract from high bits to low bits
          int shift   = 28 - (chunk_in_uint32 * BITS_PER_CHUNK);  // 28, 24, 20, 16, 12, 8, 4, 0
          int pattern = (short_code_chunk >> shift) & 0xF;        // Extract 4 bits

          // Look up in LUT
          uint32_t lut_chunk_idx = uint32_idx * chunks_per_uint32 + chunk_in_uint32;
          uint32_t lut_offset    = lut_chunk_idx * LUT_SIZE + pattern;

          // Accumulate inner product
          //                    ip += __bfloat162float(shared_lut_bf16[lut_offset]);
          ip += shared_lut[lut_offset];
        }
      }

      // Compute estimated distance
      final_1bit_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

      // Check threshold
      if (final_1bit_dist < threshold) {
        is_candidate   = true;
        final_1bit_pid = params.d_pids[cluster_start_index + vec_idx];
      }
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);

      if (candidate_slot < params.max_candidates_per_pair) {
        // Use ip slot to store distance
        shared_candidate_ips[candidate_slot] = final_1bit_dist;
        // Use index slot to store pid
        shared_candidate_indices[candidate_slot] = final_1bit_pid;
      }
    }
  }
  __syncthreads();

  // Step 2: Sort 1 bit dist directly and return results

  __syncthreads();
  //    __shared__ int probe_slot;
  if (num_candidates > 0) {
    __shared__ int probe_slot;
    {
      QueueT queue(params.topk);

      const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

      // Each warp processes different candidates
      for (int c = 0; c < candidates_per_thread; ++c) {
        int cand_idx = tid + c * num_threads;

        if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
          final_1bit_dist = shared_candidate_ips[cand_idx];
          final_1bit_pid  = shared_candidate_indices[cand_idx];
        } else {
          final_1bit_dist = INFINITY;
          final_1bit_pid  = 0;
        }
        queue.add(final_1bit_dist, final_1bit_pid);
      }

      __syncthreads();

      // Step 3: Merge results and write back top-k
      if constexpr (SharedQueueBuffer) {
        queue.done((uint8_t*)shared_lut);
      } else {
        queue.done();
      }

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

    float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      uint32_t output_offset =
        query_idx * (params.topk * params.nprobe) +
        probe_slot * params.topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (uint32_t i = 0; i < params.topk; i++) {
        float dist = params.d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
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

// instantiate the kernel with a queue type that satisfies the capacity requirement
template <int Capacity = MAX_TOP_K_WARP_SORT>
auto instantiateComputeInnerProductsWithLUTNoEX(const uint32_t topk)
{
  static_assert(Capacity <= MAX_TOP_K_WARP_SORT,
                "Capacity exceeds maximum supported: " STR(MAX_TOP_K_WARP_SORT));
  if constexpr (Capacity <=
                MAX_TOP_K_BLOCK_SORT) {  // base case: use block_sort (requires shared mem)
    using block_sort_t = typename raft::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K_BLOCK_SORT, true, T, IdxT>::type;
    return computeInnerProductsWithLUTNoEX<block_sort_t, /* SharedQueueBuffer = */ true>;
  } else {
    if (topk * WARP_SORT_CAPACITY_FACTOR >
        Capacity) {  // base case: use warp_sort (does not require shared mem)
      using warp_sort_t =
        raft::matrix::detail::select::warpsort::warp_sort_filtered<Capacity, true, T, IdxT>;
      return computeInnerProductsWithLUTNoEX<warp_sort_t, /* SharedQueueBuffer = */ false>;
    } else {  // recursive case
      return instantiateComputeInnerProductsWithLUTNoEX<Capacity / WARP_SORT_CAPACITY_FACTOR>(topk);
    }
  }
}

void SearcherGPU::SearchClusterQueryPairs(const IVFGPU& cur_ivf,
                                          IVFGPU::GPUClusterMeta* d_cluster_meta,
                                          ClusterQueryPair* d_sorted_pairs,
                                          size_t num_queries,
                                          const float* d_query,
                                          const float* d_G_k1xSumq,
                                          const float* d_G_kbxSumq,
                                          size_t nprobe,
                                          size_t topk,
                                          float* d_topk_dists,  // sizeof(float)*topk*nprobe*query
                                          PID* d_topk_pids,
                                          float* d_final_dists,
                                          PID* d_final_pids)
{
  RAFT_EXPECTS(topk <= MAX_TOP_K_WARP_SORT,
               "Top-K value exceeds maximum supported: " STR(MAX_TOP_K_WARP_SORT));

  // First allocate space for LUT
  size_t lut_size =
    num_queries * (cur_ivf.get_num_padded_dim() / BITS_PER_CHUNK) * LUT_SIZE * sizeof(float);
  // each line's space is (cur_ivf.get_num_padded_dim() / BITS_PER_CHUNK) * LUT_SIZE *
  // sizeof(float);
  float* d_lut_for_queries = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_lut_for_queries, lut_size, stream_));
  thrust::fill(thrust::cuda::par.on(stream_),
               d_lut_for_queries,
               d_lut_for_queries + (lut_size / sizeof(float)),
               -std::numeric_limits<float>::infinity());  // initially set to INVALID value
  // precompute LUTS
  launchPrecomputeLUTs(d_query, d_lut_for_queries, num_queries, D, stream_);
  //  Clean the input distances
  size_t candidates_per_query = nprobe * topk;
  size_t total_elements       = num_queries * candidates_per_query;
  int threads                 = 256;
  int blocks                  = (total_elements + threads - 1) / threads;

  initDistancesKernel<<<blocks, threads, 0, stream_>>>(d_topk_dists, total_elements);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  int* d_query_write_counters;  // One counter per query, indicates where to store final results
                                // (0~nprobe)
  RAFT_CUDA_TRY(cudaMallocAsync(&d_query_write_counters, num_queries * sizeof(int), stream_));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_query_write_counters, 0, num_queries * sizeof(int), stream_));  // Initialize to 0

  float* d_topk_threshold_batch;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_topk_threshold_batch, sizeof(float) * num_queries, stream_));
  thrust::fill(thrust::cuda::par.on(stream_),
               d_topk_threshold_batch,
               d_topk_threshold_batch + num_queries,
               std::numeric_limits<float>::infinity());
  // Then launch kernel for computation
  size_t num_pairs = num_queries * nprobe;
  uint32_t gridDim{static_cast<uint32_t>(num_pairs)};
  uint32_t blockDim{topk > MAX_TOP_K_BLOCK_SORT ? static_cast<uint32_t>(WARP_SIZE) : 256};
  size_t num_chunks = D / BITS_PER_CHUNK;
  size_t candidate_storage =
    cur_ivf.get_max_cluster_length() * (2 * sizeof(float) + sizeof(int));  // ip, idx
  size_t query_storage = D * sizeof(float);  // For shared query vector
  const int queue_buffer_smem_bytes =
    (topk > MAX_TOP_K_BLOCK_SORT)
      ? 0
      : raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(
          blockDim / WARP_SIZE, MAX_TOP_K_BLOCK_SORT);

  ComputeInnerProductsKernelParams kernelParams;
  kernelParams.d_sorted_pairs          = d_sorted_pairs;
  kernelParams.d_query                 = d_query;
  kernelParams.d_short_data            = cur_ivf.get_short_data_device();
  kernelParams.d_cluster_meta          = d_cluster_meta;
  kernelParams.d_lut_for_queries_float = d_lut_for_queries;
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
  kernelParams.max_candidates_per_pair = cur_ivf.get_max_cluster_length();
  kernelParams.ex_bits                 = cur_ivf.get_ex_bits();
  kernelParams.d_long_code             = cur_ivf.get_long_code_device();
  kernelParams.d_ex_factor  = reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device());
  kernelParams.d_pids       = cur_ivf.get_ids_device();
  kernelParams.d_topk_dists = d_topk_dists;
  kernelParams.d_topk_pids  = d_topk_pids;
  kernelParams.d_query_write_counters = d_query_write_counters;

  if (cur_ivf.get_ex_bits() != 0) {
    auto kernel            = instantiateComputeInnerProductsWithLUT(topk);
    size_t shared_mem_size = num_chunks * LUT_SIZE * sizeof(float) + candidate_storage +
                             query_storage + queue_buffer_smem_bytes;
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
  } else {
    auto kernel            = instantiateComputeInnerProductsWithLUTNoEX(topk);
    size_t shared_mem_size = max(num_chunks * LUT_SIZE * sizeof(float) +
                                   cur_ivf.get_max_cluster_length() * (sizeof(float) + sizeof(int)),
                                 (size_t)queue_buffer_smem_bytes);
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // merge results from different blocks
  mergeClusterTopKFinal(d_topk_dists,
                        d_topk_pids,
                        d_final_dists,
                        d_final_pids,
                        num_queries,
                        nprobe,
                        topk,
                        handle_,
                        /* sorted = */ false);

  RAFT_CUDA_TRY(cudaFreeAsync(d_topk_threshold_batch, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_lut_for_queries, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_query_write_counters, stream_));

  raft::resource::sync_stream(handle_);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
