/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 8/24/25.
//

#include <cuvs/neighbors/ivf_rabitq/gpu_index/searcher_gpu.cuh>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/matrix/select_k.cuh>
#include <raft/neighbors/detail/ivf_flat_interleaved_scan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

namespace cuvs::neighbors::ivf_rabitq::detail {

#define MAX_TOP_K               64  // power of 2, as local_topk_capacity, assumes that topk is less than 100
#define MAX_CANDIDATES_PER_PAIR 1000  // suppose topk = 100, M = 10

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16
static constexpr int WARP_SIZE      = 32;

// --- Tunables ---
// static constexpr int MAX_TOP_K = 1024;      // Capacity (power of two, <= 256)
static constexpr bool ASCENDING = true;  // true = keep smallest
using T                         = float;
using IdxT                      = uint32_t;

// using lut_dtype = __nv_bfloat16;  // Or use __half for FP16
using lut_dtype = __half;  // FP16 alternative

// Alias to the block_sort type you’re using:
using block_sort_t =
  typename raft::neighbors::ivf_flat::detail::flat_block_sort<MAX_TOP_K, ASCENDING, T, IdxT>::type;
// using block_sort_t_small = typename
// raft::neighbors::ivf_flat::detail::flat_block_sort<64, ASCENDING, T, IdxT>::type;

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
                          cudaStream_t stream = 0)
{
  // Initialize all LUTs to invalid value first
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;
  const size_t total_lut_size     = num_queries * lut_per_query_size;

  // Optional: Initialize to -infinity to mark as uncomputed
  // (You can skip this if you always call precompute before main kernel)
  float neg_inf = -std::numeric_limits<float>::infinity();
  cudaMemsetAsync(
    d_lut_for_queries, *reinterpret_cast<int*>(&neg_inf), total_lut_size * sizeof(float), stream);

  // Launch precompute kernel
  dim3 gridDim(num_queries, 1, 1);
  dim3 blockDim(256, 1, 1);  // Can tune this

  precomputeAllLUTs<<<gridDim, blockDim, 0, stream>>>(d_query, d_lut_for_queries, num_queries, D);

  // Check for errors
  //    CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void precomputeAllLUTs_optimized(const float* d_query,
                                            float* d_lut_for_queries,
                                            size_t num_queries,
                                            size_t D)
{
  // Each block handles one query
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Shared memory for query vector chunk and temporary LUT storage
  extern __shared__ float shared_mem[];
  float* shared_query = shared_mem;
  float* shared_lut   = shared_mem + BITS_PER_CHUNK;  // Assuming BITS_PER_CHUNK = 4

  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  float* query_lut       = d_lut_for_queries + query_idx * lut_per_query_size;
  const float* query_vec = d_query + query_idx * D;

  // Process each chunk
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    size_t dim_start = chunk_idx * BITS_PER_CHUNK;

    // Step 1: Load query chunk into shared memory (coalesced read)
    if (tid < BITS_PER_CHUNK && dim_start + tid < D) {
      shared_query[tid] = query_vec[dim_start + tid];
    }
    __syncthreads();

    // Step 2: Each thread computes multiple LUT entries
    // Distribute LUT entries across threads for coalesced writes
    for (int lut_entry = tid; lut_entry < LUT_SIZE; lut_entry += num_threads) {
      float sum = 0.0f;

      // Compute using shared memory (no global memory access)
      for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
        if (dim_start + bit_idx < D) {
          if (lut_entry & (1 << (BITS_PER_CHUNK - 1 - bit_idx))) { sum += shared_query[bit_idx]; }
        }
      }

      // Store in shared memory first
      if (lut_entry < LUT_SIZE) { shared_lut[lut_entry] = sum; }
    }
    __syncthreads();

    // Step 3: Coalesced write from shared to global memory
    // Now consecutive threads write to consecutive locations
    size_t base_offset = chunk_idx * LUT_SIZE;
    for (int i = tid; i < LUT_SIZE; i += num_threads) {
      query_lut[base_offset + i] = shared_lut[i];
    }
    __syncthreads();
  }
}

// Updated launch function
void launchPrecomputeLUTs_optimized(const float* d_query,
                                    float* d_lut_for_queries,
                                    size_t num_queries,
                                    size_t D,
                                    cudaStream_t stream = 0)
{
  dim3 gridDim(num_queries, 1, 1);
  dim3 blockDim(256, 1, 1);  // Or 128/64 depending on shared memory usage

  // Calculate shared memory size
  size_t shared_mem_size = (BITS_PER_CHUNK + LUT_SIZE) * sizeof(float);

  precomputeAllLUTs_optimized<<<gridDim, blockDim, shared_mem_size, stream>>>(
    d_query, d_lut_for_queries, num_queries, D);
}
__global__ void computeInnerProductsWithLUT(
  const ClusterQueryPair* d_sorted_pairs,
  const float* d_query,
  const uint32_t* d_short_data,
  const IVFGPU::GPUClusterMeta* d_cluster_meta,
  float* d_lut_for_queries,
  const float* d_short_factors,       // NEW
  const float* d_G_k1xSumq,           // NEW
  const float* d_G_kbxSumq,           // NEW (not used yet)
  const float* d_centroid_distances,  // NEW
                                      //        float* d_distances,  // output distances
  size_t topk,
  size_t num_queries,
  size_t nprobe,
  size_t num_pairs,
  size_t num_centroids,
  size_t D,
  const float* d_threshold,  // NEW: threshold for each query
  //        float* d_ip_results,                 // NEW: store inner products for candidates
  size_t M,                        // NEW: multiplier for topk
  size_t max_candidates_per_pair,  // NEW: max storage per pair, 1000 suggested
  size_t ex_bits,                  // NEW: bits per dimension in ex codes
  const uint8_t* d_long_code,      // NEW: long codes for all vectors
  const float* d_ex_factor,        // NEW: ex factors for distance computation
  const PID* d_pids,               // NEW: PIDs for all vectors
  float* d_topk_dists,             // NEW: output top-k distances
  PID* d_topk_pids,                // NEW: output top-k PIDs
  int* d_query_write_counters)
{
  // Each block handles one <cluster, query> pair
  //    const int block_id = blockIdx.x + blockIdx.y * gridDim.x +
  //                         blockIdx.z * gridDim.x * gridDim.y;
  const int block_id = blockIdx.x;  // simply use 1-D block

  if (block_id >= num_pairs) return;

  // Get the cluster-query pair for this block
  ClusterQueryPair pair = d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  // Check bounds
  if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

  // Get cluster metadata
  //    IVFGPU::GPUClusterMeta cluster_meta = d_cluster_meta[cluster_idx];
  size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = d_cluster_meta[cluster_idx].start_index;

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Preparation completed!\n"); }
#endif

  // Calculate LUT parameters
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Shared memory for LUT
  extern __shared__ __align__(256) float shared_lut[];

  // Thread index within the block
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointer to this query's LUT in global memory
  float* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

  // Pointer to this query's vector
  const float* query_vec = d_query + query_idx * D;
  // ------

  //    // Step 1: Check if LUT needs to be computed and compute if necessary
  //    bool need_compute_lut = false;
  //
  //    // First thread checks if LUT is invalid
  //    if (tid == 0) {
  //        if (query_lut[0] == -std::numeric_limits<float>::infinity()) {
  //            need_compute_lut = true;
  //        }
  //    }
  //
  //    // Broadcast the decision to all threads in the block
  //    __shared__ bool shared_need_compute;
  //    if (tid == 0) {
  //        shared_need_compute = need_compute_lut;
  //    }
  //    __syncthreads();
  //    need_compute_lut = shared_need_compute;
  //
  //    // If LUT needs to be computed, compute it
  //    if (need_compute_lut) {
  //        // Each thread computes part of the LUT
  //        for (size_t chunk_idx = tid; chunk_idx < num_chunks; chunk_idx += num_threads) {
  //            size_t dim_start = chunk_idx * BITS_PER_CHUNK;
  //
  //            // Compute LUT entries for this chunk
  //            for (int lut_entry = 0; lut_entry < LUT_SIZE; lut_entry++) {
  //                float sum = 0.0f;
  //
  //                // For each bit in the 4-bit pattern
  //                for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
  //                    size_t dim = dim_start + bit_idx;
  //                    if (dim < D) {  // Check if within actual dimension
  //                        // Check if bit is set in the pattern
  //                        if (lut_entry & (1 << (BITS_PER_CHUNK - 1 - bit_idx))) {
  //                            sum += query_vec[dim];
  //                        }
  //                    }
  //                }
  //
  //                // Store in global LUT
  //                size_t lut_offset = chunk_idx * LUT_SIZE + lut_entry;
  //                query_lut[lut_offset] = sum;
  //            }
  //        }
  //
  //        // Ensure all threads have finished computing their part of the LUT
  //        __syncthreads();
  //    }

  // Then Load LUT into shared memory
  // Each thread loads part of the LUT
  for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut[i] = query_lut[i];
  }

  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("LUT computation & load finished!\n"); }
#endif

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
    q_g_add   = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error = sqrtf(q_g_add);

    // Get query factor
    q_k1xsumq      = d_G_k1xSumq[query_idx];
    threshold      = d_threshold[query_idx];  // NEW: load threshold
    num_candidates = 0;                       // NEW: initialize counter
  }
  __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
//    if ( threadIdx.x == 0 ) {
//        printf("query idx: %d, threshold after loading: %f\n", query_idx, threshold);
//    }
#endif

  // Allocate shared memory for candidate storage (after LUT)
  // Assuming extern shared memory is large enough
  float* shared_candidate_dists = shared_lut + (num_chunks * LUT_SIZE);
  float* shared_candidate_ips   = shared_candidate_dists + max_candidates_per_pair;
  int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);
  int* shared_buffer            = shared_candidate_indices + max_candidates_per_pair;

  // Calculate short code parameters
  const size_t short_code_length = D / 32;               // number of uint32_t per vector
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
      //            size_t factor_offset = (cluster_start_index + vec_idx) * 3;
      //            float f_add = d_short_factors[factor_offset];
      //            float f_rescale = d_short_factors[factor_offset + 1];
      //            float f_error = d_short_factors[factor_offset + 2];
      // vec load for short factors
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
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
        uint32_t short_code_chunk = d_short_data[short_code_offset];

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
      //            constexpr float threshold_factor = 1.05;
      //            if (1) {
      if (low_dist < threshold) {
        is_candidate   = true;
        local_low_dist = est_dist;
#ifdef DEBUG_BATCH_SEARCH
        if (local_low_dist < 0) { printf("local_low_dist = %f < 0!\n", local_low_dist); }
#endif
        local_ip = ip;
      }

#ifdef DEBUG_BATCH_SEARCH
//            if ( threadIdx.x == 0 ) {
//                printf("low_dist: %f, threshold: %f\n", low_dist, threshold);
//                if (low_dist > 1000) {
//                    printf("f_add: %f, q_g_add: %f, f_rescale: %f, ip: %f, q_k1xsumq: %f,
//                    est_dist: %f\n",f_add, q_g_add, f_rescale, ip, q_k1xsumq, est_dist);
//                }
//            }
#endif
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
#ifdef DEBUG_BATCH_SEARCH
      if (threadIdx.x == 10) {
        //                printf("num_candidates: %d\n", num_candidates);
      }
#endif
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_dists[candidate_slot]   = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

  //// -----------
  //
  //    // Step 2 Part 1: Another option: Compute distances using direct inner product
  //
  //    // Shared values for this <cluster, query> pair
  //    __shared__ float q_g_add;      // squared distance to centroid
  //    __shared__ float q_k1xsumq;    // query factor
  //    __shared__ float q_g_error;    // sqrt(q_g_add)
  //    __shared__ float threshold;     // threshold for this query
  //    __shared__ int num_candidates;  // counter for candidates
  //
  //    // Load shared query-cluster values
  //    if (tid == 0) {
  //        // Get squared distance from query to this cluster's centroid
  //        q_g_add = d_centroid_distances[query_idx * num_centroids + cluster_idx];
  //        q_g_error = sqrtf(q_g_add);
  //
  //        // Get query factor
  //        q_k1xsumq = d_G_k1xSumq[query_idx];
  //        threshold = d_threshold[query_idx];
  //        num_candidates = 0;
  //    }
  //    __syncthreads();
  //
  //    // Load query vector into shared memory for direct computation
  //    float* shared_query = shared_lut;  // Reuse LUT space for query vector
  //    for (size_t i = tid; i < D; i += num_threads) {
  //        shared_query[i] = d_query[query_idx * D + i];
  //    }
  //    __syncthreads();
  //
  //    // Allocate shared memory for candidate storage (after query vector)
  //    float* shared_candidate_dists = shared_query + D;
  //    float* shared_candidate_ips = shared_candidate_dists + max_candidates_per_pair;
  //    int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);
  //    int* shared_buffer = shared_candidate_indices + max_candidates_per_pair;
  //
  //    // Calculate short code parameters
  //    const size_t short_code_length = D / 32;  // number of uint32_t per vector
  //
  //    // Each thread processes one or more vectors
  //    const int vectors_per_iteration = num_threads;
  //
  //    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base +=
  //    vectors_per_iteration) {
  //        size_t vec_idx = vec_base + tid;
  //
  //        float local_low_dist = INFINITY;
  //        float local_ip = 0.0f;
  //        bool is_candidate = false;
  //
  //        if (vec_idx < num_vectors_in_cluster) {
  //            // Load short factors for this vector
  //            size_t factor_offset = (cluster_start_index + vec_idx) * 3;
  //            float f_add = d_short_factors[factor_offset];
  //            float f_rescale = d_short_factors[factor_offset + 1];
  //            float f_error = d_short_factors[factor_offset + 2];
  //
  //            // Compute inner product directly
  //            float ip = 0.0f;
  //
  //            // Process each uint32_t of the short code
  //            for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
  //                // Access short code in transposed layout
  //                size_t short_code_offset = cluster_start_index * short_code_length +
  //                                           uint32_idx * num_vectors_in_cluster +
  //                                           vec_idx;
  //                uint32_t short_code_chunk = d_short_data[short_code_offset];
  //
  //                // Process each bit in the uint32_t
  //                // Remember: lowest dim is at bit 31 (MSB), highest dim at bit 0
  //                for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
  //                    // Calculate the actual dimension index
  //                    size_t dim = uint32_idx * 32 + bit_idx;
  //
  //                    // Extract the bit (from MSB to LSB)
  //                    int bit_position = 31 - bit_idx;
  //                    bool bit_value = (short_code_chunk >> bit_position) & 0x1;
  //
  //                    // If bit is 1, add the query value; if 0, add nothing
  //                    if (bit_value) {
  //                        ip += shared_query[dim];
  //                    }
  //                }
  //            }
  //
  //            // Compute estimated distance
  //            float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
  //
  //            // Compute lower bound
  //            float low_dist = est_dist - f_error * q_g_error;
  //
  //            // Check threshold
  //            if (low_dist < threshold) {
  //                is_candidate = true;
  //                local_low_dist = low_dist;
  //                local_ip = ip;
  //            }
  //        }
  //
  //        // Collectively add candidates to shared memory
  //        __syncwarp();  // Sync within warp for atomics
  //
  //        if (is_candidate) {
  //            int candidate_slot = atomicAdd(&num_candidates, 1);
  //            if (candidate_slot < max_candidates_per_pair) {
  //                shared_candidate_dists[candidate_slot] = local_low_dist;
  //                shared_candidate_ips[candidate_slot] = local_ip;
  //                shared_candidate_indices[candidate_slot] = vec_idx;
  //            }
  //        }
  //    }
  //    __syncthreads();
  // ------

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("1bit estimated distance computation finished!\n");
  }
#endif

  // Step 2 Part 2: Determine which candidates to use
  int final_num_candidates = min(num_candidates, (int)max_candidates_per_pair);
//    size_t topk_threshold = topk * M;
//    size_t topk_threshold = num_candidates;
#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("final_num_candidates_before: %d\n", final_num_candidates);
  }
#endif
  // Check if we need to sort and select top-k*M
//    if (final_num_candidates > topk_threshold) {
//        // Sort candidates by low_dist in shared memory
//        // Using simple bitonic sort for now (can be optimized)
//
//        // Parallel bitonic sort in shared memory
//        for (int size = 2; size <= final_num_candidates; size *= 2) {
//            for (int stride = size / 2; stride > 0; stride /= 2) {
//                __syncthreads();
//
//                for (int idx = tid; idx < final_num_candidates; idx += num_threads) {
//                    int partner = idx ^ stride;
//
//                    if (partner > idx && partner < final_num_candidates) {
//                        if ((idx & size) == 0) {  // Ascending
//                            if (shared_candidate_dists[idx] > shared_candidate_dists[partner]) {
//                                // Swap distances
//                                float temp_dist = shared_candidate_dists[idx];
//                                shared_candidate_dists[idx] = shared_candidate_dists[partner];
//                                shared_candidate_dists[partner] = temp_dist;
//
//                                // Swap IPs
//                                float temp_ip = shared_candidate_ips[idx];
//                                shared_candidate_ips[idx] = shared_candidate_ips[partner];
//                                shared_candidate_ips[partner] = temp_ip;
//
//                                // Swap indices
//                                int temp_idx = shared_candidate_indices[idx];
//                                shared_candidate_indices[idx] = shared_candidate_indices[partner];
//                                shared_candidate_indices[partner] = temp_idx;
//                            }
//                        } else {  // Descending
//                            if (shared_candidate_dists[idx] < shared_candidate_dists[partner]) {
//                                // Swap distances
//                                float temp_dist = shared_candidate_dists[idx];
//                                shared_candidate_dists[idx] = shared_candidate_dists[partner];
//                                shared_candidate_dists[partner] = temp_dist;
//
//                                // Swap IPs
//                                float temp_ip = shared_candidate_ips[idx];
//                                shared_candidate_ips[idx] = shared_candidate_ips[partner];
//                                shared_candidate_ips[partner] = temp_ip;
//
//                                // Swap indices
//                                int temp_idx = shared_candidate_indices[idx];
//                                shared_candidate_indices[idx] = shared_candidate_indices[partner];
//                                shared_candidate_indices[partner] = temp_idx;
//                            }
//                        }
//                    }
//                }
//            }
//        }
//
//        __syncthreads();
//
//        // Keep only top-k*M
//        final_num_candidates = topk_threshold;
//    }
#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Sorting TOPK*M finished!\n"); }
#endif
  // -----------------------
  //    // if only use 1bit code:
  //
  //    if (final_num_candidates > 0) {
  //        using block_sort_t = typename raft::neighbors::ivf_flat::detail::flat_block_sort<
  //                MAX_TOP_K, true, float, uint32_t>::type;
  //        block_sort_t queue(topk);
  //
  //
  //
  //        const int adds_per_thread = (final_num_candidates + num_threads - 1) / num_threads;
  //
  //        for (int round = 0; round < adds_per_thread; round++) {
  //            int cand_idx = tid + round * num_threads;
  //
  //            float ex_dist;
  //            uint32_t pid;
  //
  //            if (cand_idx < final_num_candidates) {
  //                // Get pre-computed values
  //
  //                int local_vec_idx = shared_candidate_indices[cand_idx];
  //                size_t global_vec_idx = cluster_start_index + local_vec_idx;
  //
  //
  //
  //                // Compute final distance using pre-computed ip2
  //                ex_dist = shared_candidate_dists[cand_idx];
  //
  //                // Get PID
  //                pid = (uint32_t)d_pids[global_vec_idx];
  //            } else {
  //                // Thread has no valid candidate for this round - use dummy values
  //                ex_dist = INFINITY;
  //                pid = 0;
  //            }
  //
  //            // ALL threads call queue.add() exactly once per round
  //            queue.add(ex_dist, pid);
  //        }
  //
  //        __syncthreads();
  //
  //
  //        uint8_t* queue_buffer = (uint8_t*)shared_buffer;
  //        queue.done(queue_buffer);
  //
  //        // Atomically get write position
  //        __shared__ int probe_slot;
  //        if (tid == 0) {
  //            probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1);
  //        }
  //        __syncthreads();
  //
  //        if (probe_slot >= nprobe) {
  //            return;
  //        }
  //
  //        // Calculate output offset and store results
  //        size_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
  //        queue.store(d_topk_dists + output_offset,
  //                    (uint32_t*)(d_topk_pids + output_offset));
  //
  //
  //
  //

  //------
  //// Step 3: Compute more accurate distances and select top-k
  //
  //    // Initialize the block-level top-k queue
  //    __syncthreads();
  // #ifdef DEBUG_BATCH_SEARCH
  //    if (blockIdx.x == 0  && threadIdx.x == 0 ) {
  //        printf("final_num_candidates_after: %d\n", final_num_candidates);
  //    }
  // #endif
  //    if (final_num_candidates > 0) {
  //        using block_sort_t = typename raft::neighbors::ivf_flat::detail::flat_block_sort<
  //                MAX_TOP_K, true, float, uint32_t>::type;
  //        block_sort_t queue(topk);
  //
  //        // Additional shared values needed for Step 3
  //        __shared__ float q_kbxsumq;
  //        if (tid == 0) {
  //            q_kbxsumq = d_G_kbxSumq[query_idx];
  //        }
  //        __syncthreads();
  //
  //        // Calculate long code parameters
  //        const size_t long_code_size = (D * ex_bits + 7) / 8;
  //
  //        // Load query vector to shared memory for efficient access
  //        float* shared_query = (float*) (shared_lut);
  ////    shared_query = (float*)(shared_lut);
  //
  //        // Cooperatively load query vector to shared memory
  //        for (size_t i = tid; i < D; i += num_threads) {
  //            shared_query[i] = d_query[query_idx * D + i];
  //        }
  //        __syncthreads();
  //
  //        // Step 3 Part 1: CORRECTED - Each THREAD processes different candidates
  //        // Process candidates using thread-level parallelism
  //        for (int cand_idx = tid; cand_idx < final_num_candidates; cand_idx += num_threads) {
  //            // Get candidate information from shared memory
  //            float ip = shared_candidate_ips[cand_idx];
  //            int local_vec_idx = shared_candidate_indices[cand_idx];
  //            size_t global_vec_idx = cluster_start_index + local_vec_idx;
  //
  //            // Load ex factors for this vector
  //            size_t ex_factor_offset = global_vec_idx * 2;
  //            float f_ex_add = d_ex_factor[ex_factor_offset];
  //            float f_ex_rescale = d_ex_factor[ex_factor_offset + 1];
  //
  //            // Pointer to this vector's long code
  //            const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;
  //
  //            // Compute ip2 - single thread computes full inner product
  //            float ip2 = 0.0f;
  //            for (size_t d = 0; d < D; d++) {
  //                uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
  //                float ex_val = (float) code_val;
  //                ip2 += shared_query[d] * ex_val;
  //            }
  //
  //            // Compute final distance
  //            float ex_dist = f_ex_add + q_g_add +
  //                            (f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 +
  //                            q_kbxsumq));
  //
  // #ifdef DEBUG_BATCH_SEARCH
  //            // use 1-bit codes to check
  ////        ex_dist = shared_candidate_dists[cand_idx];
  ////            if (ex_dist < shared_candidate_dists[cand_idx]) {
  ////                printf("Error, ex_dist lower than low_dist! ex_dist: %f, low_dist: %f\n",
  ////                       ex_dist, shared_candidate_dists[cand_idx]);
  ////            }
  //
  // #endif
  //
  // #ifdef DEBUG_BATCH_SEARCH
  ////        if ( threadIdx.x == 0 ) {
  ////            printf("ex_dist: %f\n", ex_dist);
  ////        }
  ////        if ( query_idx == 0 ) {
  ////            printf("ex_dist: %f\n", ex_dist);
  ////        }
  // #endif
  //             // Get PID
  //             uint32_t pid = (uint32_t) d_pids[global_vec_idx];
  //
  //             // ALL threads call queue.add()
  //             queue.add(ex_dist, pid);
  // #ifdef DEBUG_BATCH_SEARCH
  //             if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
  //                 printf("Wrong pid/ex_dist!!!! PID: %d, ex_dist: %f, query_idx: %d\n",pid,
  //                 ex_dist, query_idx);
  //             }
  // #endif
  //         }
  //
  //         // CRITICAL: Threads without candidates must still participate
  //         // Add dummy values for proper warp synchronization
  //         const int remainder = final_num_candidates % num_threads;
  //         if (remainder != 0 && tid >= remainder) {
  //             // These threads don't have real candidates in the last iteration
  //             // Add dummy values that won't be selected
  //             queue.add(INFINITY, 0);
  ////            auto a = INFINITY;
  ////            printf("INFINITY: %f\n", a);
  //        }
  //
  //        __syncthreads();
  //
  //        // Step 3 Part 2: Merge results and write back top-k
  //
  //        // Reuse LUT space as shared memory buffer for queue.done()
  //        // Note: May encounter issues if reuse LUT (when need buffer size can be larger than
  //        LUT) uint8_t* queue_buffer = (uint8_t*) shared_buffer;
  //
  //        // Merge results from different warps
  //        queue.done(reinterpret_cast<uint8_t*>(queue_buffer));
  //
  //        // Atomically get the next write position for this query
  //        // This returns the old value and increments the counter
  //        // storing block wise, thus only 1 thread do the atomic add
  //        __shared__ int probe_slot;
  //        if (tid == 0) {
  //            probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1);
  //        }
  //
  //        __syncthreads();
  // #ifdef DEBUG_BATCH_SEARCH
  //        if (blockIdx.x == 0 && threadIdx.x == 0) {
  //            printf("proble_slot: %d\n", probe_slot);
  //        }
  // #endif
  //        // Check if we're within bounds (safety check)
  //        if (probe_slot >= nprobe) {
  //            // This shouldn't happen if pairs are set up correctly
  //            return;
  //        }
  //
  //        // Calculate output offset using the atomic slot
  //        size_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
  //
  //        // Store the top-k results at the atomically determined position
  //        queue.store(d_topk_dists + output_offset,
  //                    (uint32_t*) (d_topk_pids + output_offset));
  //--------

  // Step 3 opt: Compute more accurate distances and select top-k
  // Opt: warp-level dist and then thread-level ex dist restore

  __syncthreads();
  //    __shared__ int probe_slot;
  if (final_num_candidates > 0) {
    using block_sort_t = typename raft::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
    block_sort_t queue(topk);

    // Additional shared values needed for Step 3
    __shared__ float q_kbxsumq;
    if (tid == 0) { q_kbxsumq = d_G_kbxSumq[query_idx]; }
    __syncthreads();

    // Calculate long code parameters
    const size_t long_code_size = (D * ex_bits + 7) / 8;

    // Load query vector to shared memory
    float* shared_query = (float*)(shared_lut);
    for (size_t i = tid; i < D; i += num_threads) {
      shared_query[i] = d_query[query_idx * D + i];
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
      const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;

      // Warp-level IP2 computation
      float ip2 = 0.0f;

      // Each thread in warp processes different dimensions
      for (size_t d = lane_id; d < D; d += WARP_SIZE) {
        // Extract ex_bits value for this dimension
        uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
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
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part1 finished!\n"); }
#endif

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
#ifdef DEBUG_BATCH_SEARCH
        if (local_vec_idx > num_vectors_in_cluster) {
          printf("Error! local_vec_index %d moare than num_vectors %d in cluster!\n",
                 local_vec_idx,
                 num_vectors_in_cluster);
        }
#endif

        // Load ex factors for this vector
        //                size_t ex_factor_offset = global_vec_idx * 2;
        //                float f_ex_add = d_ex_factor[ex_factor_offset];
        //                float f_ex_rescale = d_ex_factor[ex_factor_offset + 1];
        // vec load version
        float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
        float f_ex_add     = ex_factors.x;
        float f_ex_rescale = ex_factors.y;

        // Compute final distance using pre-computed ip2
        ex_dist = f_ex_add + q_g_add +
                  f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
//                ex_dist = ex_dist+1;
#ifdef DEBUG_BATCH_SEARCH
        if (ex_dist < 0 && cand_idx < final_num_candidates) {
          printf(
            "ex_dist: %f, f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f， pos %d in cluster %d\n",
            ex_dist,
            f_ex_add,
            f_ex_rescale,
            ip,
            ip2,
            local_vec_idx,
            cluster_idx);
          if (cand_idx + 1 < final_num_candidates) {
            printf("next_data's f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f\n",
                   d_ex_factor[global_vec_idx * 2 + 2],
                   d_ex_factor[global_vec_idx * 2 + 3],
                   shared_candidate_ips[cand_idx + 1],
                   shared_ip2_results[cand_idx + 1]);
          }
        }
#endif
        // Get PID
        pid = (uint32_t)d_pids[global_vec_idx];

      } else {
        // Thread has no valid candidate for this round - use dummy values
        ex_dist = INFINITY;
        pid     = 0;
      }
#ifdef DEBUG_BATCH_SEARCH
      if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
        printf(
          "Wrong pid/ex_dist! PID: %d, ex_dist: %f, ip2: %f, query_idx: %d, max_candidate_num: "
          "%ld, num_cluster_vectors: %d\n",
          pid,
          ex_dist,
          shared_candidate_dists[cand_idx],
          query_idx,
          max_candidates_per_pair,
          num_vectors_in_cluster);
        //                ex_dist = INFINITY;
      }
#endif
      // ALL threads call queue.add() exactly once per round
      queue.add(ex_dist, pid);
    }

    __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part2 finished!\n"); }
#endif

    // Step 3 Part 3: Merge results and write back top-k

    uint8_t* queue_buffer = (uint8_t*)shared_buffer;
    queue.done(queue_buffer);

    // Atomically get write position
    __shared__ int probe_slot;
    if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
    __syncthreads();

    if (probe_slot >= nprobe) {
      //            printf("Impossible!!!!!!!\n");
      return;
    }

    // Calculate output offset and store results
    size_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
    queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));

    //--------
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) {
      printf(
        "dist, pid: %f, %d\n", *(d_topk_dists + output_offset), *(d_topk_pids + output_offset));
      printf("followed dist, pid: %f, %d\n",
             *(d_topk_dists + output_offset + 1),
             *(d_topk_pids + output_offset + 1));
    }
#endif

#ifdef DEBUG_BATCH_SEARCH
    if (threadIdx.x == 0) {
      if (/*num_candidates < topk ||*/ d_topk_dists[output_offset] < 0) {
        printf("Num candidates = %d < topk = %d \n", num_candidates, topk);
        for (int i = 0; i < topk; i++) {
          printf("pair %d: dist = %f, pid = %d\n",
                 i,
                 *(d_topk_dists + output_offset + i),
                 *(d_topk_pids + output_offset + i));
        }
      }
    }
#endif
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part3 finished!\n"); }
#endif

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Final topk for the cluster get!\n"); }
#endif

    __shared__ float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      size_t output_offset = query_idx * (topk * nprobe) +
                             probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (size_t i = 0; i < topk; i++) {
        float dist = d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
      }
    }

    __syncthreads();

    // Update threshold using atomicMin (for floats)
    // max_topk_dist should be > 0 to prevent using initialized memory
    if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
      // Use integer interpretation for atomic operations
      int* threshold_ptr = (int*)(d_threshold + query_idx);
      int new_val        = __float_as_int(max_topk_dist);

      // Atomic minimum for floats (assuming positive distances)
      atomicMin(threshold_ptr, new_val);

#ifdef DEBUG_BATCH_SEARCH
//        if ( threadIdx.x == 0 ) {
//            printf("Update threshold from %f to %f!\n", threshold, max_topk_dist);
//        }
#endif
      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("TOPK threshold updated!\n"); }
#endif
  }
}

__global__ void computeInnerProductsWithLUT16(
  const ClusterQueryPair* d_sorted_pairs,
  const float* d_query,
  const uint32_t* d_short_data,
  const IVFGPU::GPUClusterMeta* d_cluster_meta,
  lut_dtype* d_lut_for_queries,
  const float* d_short_factors,       // NEW
  const float* d_G_k1xSumq,           // NEW
  const float* d_G_kbxSumq,           // NEW (not used yet)
  const float* d_centroid_distances,  // NEW
                                      //        float* d_distances,  // output distances
  size_t topk,
  size_t num_queries,
  size_t nprobe,
  size_t num_pairs,
  size_t num_centroids,
  size_t D,
  const float* d_threshold,  // NEW: threshold for each query
  //        float* d_ip_results,                 // NEW: store inner products for candidates
  size_t M,                        // NEW: multiplier for topk
  size_t max_candidates_per_pair,  // NEW: max storage per pair, 1000 suggested
  size_t ex_bits,                  // NEW: bits per dimension in ex codes
  const uint8_t* d_long_code,      // NEW: long codes for all vectors
  const float* d_ex_factor,        // NEW: ex factors for distance computation
  const PID* d_pids,               // NEW: PIDs for all vectors
  float* d_topk_dists,             // NEW: output top-k distances
  PID* d_topk_pids,                // NEW: output top-k PIDs
  int* d_query_write_counters)
{
  // Each block handles one <cluster, query> pair
  //    const int block_id = blockIdx.x + blockIdx.y * gridDim.x +
  //                         blockIdx.z * gridDim.x * gridDim.y;
  const int block_id = blockIdx.x;  // simply use 1-D block

  if (block_id >= num_pairs) return;

  // Get the cluster-query pair for this block
  ClusterQueryPair pair = d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  // Check bounds
  if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

  // Get cluster metadata
  //    IVFGPU::GPUClusterMeta cluster_meta = d_cluster_meta[cluster_idx];
  size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = d_cluster_meta[cluster_idx].start_index;

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Preparation completed!\n"); }
#endif

  // Calculate LUT parameters
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Shared memory for LUT
  extern __shared__ __align__(256) char shared_mem_raw[];
  lut_dtype* shared_lut_bf16 = reinterpret_cast<lut_dtype*>(shared_mem_raw);
  // Calculate offset for other shared arrays after BF16 LUT
  size_t lut_bytes = num_chunks * LUT_SIZE * sizeof(lut_dtype);

  // Thread index within the block
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointer to this query's LUT in global memory
  lut_dtype* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

  // Pointer to this query's vector
  //    const float* query_vec = d_query + query_idx * D;

  // Then Load LUT into shared memory
  // Direct copy of BF16 values
  for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut_bf16[i] = query_lut[i];
  }

  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("LUT computation & load finished!\n"); }
#endif

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
    q_g_add   = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error = sqrtf(q_g_add);

    // Get query factor
    q_k1xsumq      = d_G_k1xSumq[query_idx];
    threshold      = d_threshold[query_idx];  // NEW: load threshold
    num_candidates = 0;                       // NEW: initialize counter
  }
  __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
  //    if ( threadIdx.x == 0 ) {
//        printf("query idx: %d, threshold after loading: %f\n", query_idx, threshold);
//    }
#endif

  // Allocate shared memory for candidate storage (after LUT)
  // Assuming extern shared memory is large enough
  float* shared_candidate_ips   = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
  int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);
  int* shared_buffer            = shared_candidate_indices + max_candidates_per_pair;

  // Calculate short code parameters
  const size_t short_code_length = D / 32;               // number of uint32_t per vector
  const size_t chunks_per_uint32 = 32 / BITS_PER_CHUNK;  // 8 chunks per uint32_t

  // Each thread processes one or more vectors
  // We'll use a grid-stride loop to handle all vectors in the cluster
  const int vectors_per_iteration = num_threads;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    //        float local_low_dist = INFINITY;
    float local_ip    = 0.0f;
    bool is_candidate = false;

    if (vec_idx < num_vectors_in_cluster) {
      // vec load for short factors
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
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
        uint32_t short_code_chunk = d_short_data[short_code_offset];

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
          //                    ip += __bfloat162float(shared_lut_bf16[lut_offset]);
          ip += __half2float(shared_lut_bf16[lut_offset]);
        }
      }

      // Compute estimated distance
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

      // Compute lower bound
      float low_dist = est_dist - f_error * q_g_error;

      // Check threshold
      //            constexpr float threshold_factor = 1.05;
      //            if (1) {
      if (low_dist < threshold) {
        is_candidate = true;
//                local_low_dist = est_dist;
#ifdef DEBUG_BATCH_SEARCH
//                if (local_low_dist < 0) {
//                    printf ("local_low_dist = %f < 0!\n", local_low_dist);
//                }
#endif
        local_ip = ip;
      }

#ifdef DEBUG_BATCH_SEARCH
      //            if ( threadIdx.x == 0 ) {
//                printf("low_dist: %f, threshold: %f\n", low_dist, threshold);
//                if (low_dist > 1000) {
//                    printf("f_add: %f, q_g_add: %f, f_rescale: %f, ip: %f, q_k1xsumq: %f,
//                    est_dist: %f\n",f_add, q_g_add, f_rescale, ip, q_k1xsumq, est_dist);
//                }
//            }
#endif
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
#ifdef DEBUG_BATCH_SEARCH
      if (threadIdx.x == 10) {
        //                printf("num_candidates: %d\n", num_candidates);
      }
#endif
      if (candidate_slot < max_candidates_per_pair) {
        //                shared_candidate_dists[candidate_slot] = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("1bit estimated distance computation finished!\n");
  }
#endif

  // Step 2 Part 2: Determine which candidates to use
  int final_num_candidates = min(num_candidates, (int)max_candidates_per_pair);
//    size_t topk_threshold = topk * M;
//    size_t topk_threshold = num_candidates;
#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("final_num_candidates_before: %d\n", final_num_candidates);
  }
#endif

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Sorting TOPK*M finished!\n"); }
#endif

  // Step 3 opt: Compute more accurate distances and select top-k
  // Opt: warp-level dist and then thread-level ex dist restore

  __syncthreads();
  //    __shared__ int probe_slot;
  if (final_num_candidates > 0) {
    using block_sort_t = typename raft::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
    block_sort_t queue(topk);

    // Additional shared values needed for Step 3
    __shared__ float q_kbxsumq;
    if (tid == 0) { q_kbxsumq = d_G_kbxSumq[query_idx]; }
    __syncthreads();

    // Calculate long code parameters
    const size_t long_code_size = (D * ex_bits + 7) / 8;

    // Load query vector to shared memory
    float* shared_query = (float*)(shared_buffer);
    for (size_t i = tid; i < D; i += num_threads) {
      shared_query[i] = d_query[query_idx * D + i];
    }
    __syncthreads();

    // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

    // Reuse shared_candidate_dists to store IP2 results
    float* shared_ip2_results = reinterpret_cast<float*>(shared_lut_bf16);

    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;
    const int num_warps = num_threads / WARP_SIZE;

    // Each warp processes different candidates
    for (int cand_idx = warp_id; cand_idx < final_num_candidates; cand_idx += num_warps) {
      int local_vec_idx     = shared_candidate_indices[cand_idx];
      size_t global_vec_idx = cluster_start_index + local_vec_idx;

      // Pointer to this vector's long code
      const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;

      // Warp-level IP2 computation
      float ip2 = 0.0f;

      // Each thread in warp processes different dimensions
      for (size_t d = lane_id; d < D; d += WARP_SIZE) {
        // Extract ex_bits value for this dimension
        uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
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
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part1 finished!\n"); }
#endif

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
#ifdef DEBUG_BATCH_SEARCH
        if (local_vec_idx > num_vectors_in_cluster) {
          printf("Error! local_vec_index %d moare than num_vectors %d in cluster!\n",
                 local_vec_idx,
                 num_vectors_in_cluster);
        }
#endif

        // Load ex factors for this vector
        //                size_t ex_factor_offset = global_vec_idx * 2;
        //                float f_ex_add = d_ex_factor[ex_factor_offset];
        //                float f_ex_rescale = d_ex_factor[ex_factor_offset + 1];
        // vec load version
        float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
        float f_ex_add     = ex_factors.x;
        float f_ex_rescale = ex_factors.y;

        // Compute final distance using pre-computed ip2
        ex_dist = f_ex_add + q_g_add +
                  f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
//                ex_dist = ex_dist+1;
#ifdef DEBUG_BATCH_SEARCH
        if (ex_dist < 0 && cand_idx < final_num_candidates) {
          printf("f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f， pos %d in cluster %d\n",
                 f_ex_add,
                 f_ex_rescale,
                 ip,
                 ip2,
                 local_vec_idx,
                 cluster_idx);
          if (cand_idx + 1 < final_num_candidates) {
            printf("next_data's f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f\n",
                   d_ex_factor[global_vec_idx * 2 + 2],
                   d_ex_factor[global_vec_idx * 2 + 3],
                   shared_candidate_ips[cand_idx + 1],
                   shared_ip2_results[cand_idx + 1]);
          }
        }
#endif
        // Get PID
        pid = (uint32_t)d_pids[global_vec_idx];

      } else {
        // Thread has no valid candidate for this round - use dummy values
        ex_dist = INFINITY;
        pid     = 0;
      }
#ifdef DEBUG_BATCH_SEARCH
//            if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
//                printf("Wrong pid/ex_dist! PID: %d, ex_dist: %f, ip2: %f, query_idx: %d,
//                max_candidate_num: %ld, num_cluster_vectors: %d\n",
//                       pid, ex_dist,shared_candidate_dists[cand_idx], query_idx,
//                       max_candidates_per_pair, num_vectors_in_cluster);
////                ex_dist = INFINITY;
//            }
#endif
      // ALL threads call queue.add() exactly once per round
      queue.add(ex_dist, pid);
    }

    __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part2 finished!\n"); }
#endif

    // Step 3 Part 3: Merge results and write back top-k

    uint8_t* queue_buffer = (uint8_t*)shared_lut_bf16;
    queue.done(queue_buffer);

    // Atomically get write position
    __shared__ int probe_slot;
    if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
    __syncthreads();

    if (probe_slot >= nprobe) {
      //            printf("Impossible!!!!!!!\n");
      return;
    }

    // Calculate output offset and store results
    size_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
    queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));

    //--------
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) {
      printf(
        "dist, pid: %f, %d\n", *(d_topk_dists + output_offset), *(d_topk_pids + output_offset));
      printf("followed dist, pid: %f, %d\n",
             *(d_topk_dists + output_offset + 1),
             *(d_topk_pids + output_offset + 1));
    }
#endif

#ifdef DEBUG_BATCH_SEARCH
    if (threadIdx.x == 0) {
      if (/*num_candidates < topk ||*/ d_topk_dists[output_offset] < 0) {
        printf("Num candidates = %d < topk = %d \n", num_candidates, topk);
        for (int i = 0; i < topk; i++) {
          printf("pair %d: dist = %f, pid = %d\n",
                 i,
                 *(d_topk_dists + output_offset + i),
                 *(d_topk_pids + output_offset + i));
        }
      }
    }
#endif
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part3 finished!\n"); }
#endif

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Final topk for the cluster get!\n"); }
#endif

    __shared__ float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      size_t output_offset = query_idx * (topk * nprobe) +
                             probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (size_t i = 0; i < topk; i++) {
        float dist = d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
      }
    }

    __syncthreads();

    // Update threshold using atomicMin (for floats)
    // max_topk_dist should be > 0 to prevent using initialized memory
    if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
      // Use integer interpretation for atomic operations
      int* threshold_ptr = (int*)(d_threshold + query_idx);
      int new_val        = __float_as_int(max_topk_dist);

      // Atomic minimum for floats (assuming positive distances)
      atomicMin(threshold_ptr, new_val);

#ifdef DEBUG_BATCH_SEARCH
      //        if ( threadIdx.x == 0 ) {
//            printf("Update threshold from %f to %f!\n", threshold, max_topk_dist);
//        }
#endif
      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("TOPK threshold updated!\n"); }
#endif
  }
}

// optimize loops and data types
__global__ void
//__launch_bounds__(256, 4)
computeInnerProductsWithLUT16Opt(
  const ClusterQueryPair* d_sorted_pairs,
  const float* d_query,
  const uint32_t* d_short_data,
  const IVFGPU::GPUClusterMeta* d_cluster_meta,
  lut_dtype* d_lut_for_queries,
  const float* d_short_factors,       // NEW
  const float* d_G_k1xSumq,           // NEW
  const float* d_G_kbxSumq,           // NEW (not used yet)
  const float* d_centroid_distances,  // NEW
  uint32_t topk,
  uint32_t num_queries,
  uint32_t nprobe,
  uint32_t num_pairs,
  uint32_t num_centroids,
  uint32_t D,
  const float* d_threshold,          // NEW: threshold for each query
  uint32_t M,                        // NEW: multiplier for topk
  uint32_t max_candidates_per_pair,  // NEW: max storage per pair, 1000 suggested
  uint32_t ex_bits,                  // NEW: bits per dimension in ex codes
  const uint8_t* d_long_code,        // NEW: long codes for all vectors
  const float* d_ex_factor,          // NEW: ex factors for distance computation
  const PID* d_pids,                 // NEW: PIDs for all vectors
  float* d_topk_dists,               // NEW: output top-k distances
  PID* d_topk_pids,                  // NEW: output top-k PIDs
  int* d_query_write_counters)
{
  // Each block handles one <cluster, query> pair
  //    const int block_id = blockIdx.x + blockIdx.y * gridDim.x +
  //                         blockIdx.z * gridDim.x * gridDim.y;
  const int block_id = blockIdx.x;  // simply use 1-D block

  if (block_id >= num_pairs) return;

  // Get the cluster-query pair for this block
  ClusterQueryPair pair = d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  // Check bounds
  if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

  // Get cluster metadata
  size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = d_cluster_meta[cluster_idx].start_index;

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Preparation completed!\n"); }
#endif

  // Calculate LUT parameters
  const uint32_t num_chunks         = D / BITS_PER_CHUNK;
  const uint32_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Shared memory for LUT
  extern __shared__ __align__(256) char shared_mem_raw[];
  lut_dtype* shared_lut_bf16 = reinterpret_cast<lut_dtype*>(shared_mem_raw);
  // Calculate offset for other shared arrays after BF16 LUT
  uint32_t lut_bytes = num_chunks * LUT_SIZE * sizeof(lut_dtype);

  // Thread index within the block
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointer to this query's LUT in global memory
  lut_dtype* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

  // Then Load LUT into shared memory
  // Direct copy of BF16 values
  for (uint32_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut_bf16[i] = query_lut[i];
  }

  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("LUT computation & load finished!\n"); }
#endif

  // Step 2 Part 1: Compute distances using LUT && decide candidates

  // Shared values for this <cluster, query> pair

  __shared__ int num_candidates;  // counter for candidates
  __shared__ float q_g_add;       // squared distance to centroid
  __shared__ float q_k1xsumq;     // query factor
  __shared__ float q_g_error;     // sqrt(q_g_add)
  __shared__ float threshold;     // threshold for this query
  // Load shared query-cluster values
  if (tid == 0) {
    // Get squared distance from query to this cluster's centroid
    q_g_add   = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error = sqrtf(q_g_add);

    // Get query factor
    q_k1xsumq      = d_G_k1xSumq[query_idx];
    threshold      = d_threshold[query_idx];  // NEW: load threshold
    num_candidates = 0;                       // NEW: initialize counter
  }
  __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
  //    if ( threadIdx.x == 0 ) {
//        printf("query idx: %d, threshold after loading: %f\n", query_idx, threshold);
//    }
#endif

  // Allocate shared memory for candidate storage (after LUT)
  // Assuming extern shared memory is large enough
  float* shared_candidate_ips;
  if (lut_bytes < max_candidates_per_pair * sizeof(float)) {
    shared_candidate_ips =
      reinterpret_cast<float*>(shared_mem_raw + max_candidates_per_pair * sizeof(float));
  } else {
    shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
  }
  int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);
  int* shared_buffer            = shared_candidate_indices + max_candidates_per_pair;

  // Calculate short code parameters
  const uint32_t short_code_length = D / 32;               // number of uint32_t per vector
  const uint32_t chunks_per_uint32 = 32 / BITS_PER_CHUNK;  // 8 chunks per uint32_t

  // Each thread processes one or more vectors
  // We'll use a grid-stride loop to handle all vectors in the cluster
  const int vectors_per_iteration = num_threads;
#ifdef DEBUG_BATCH_SEARCH
  if (threadIdx.x == 0 && num_vectors_in_cluster <= 0) {
    printf("Cluster %d has no vectors!\n", cluster_idx);
  }
#endif

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    float local_ip    = 0.0f;
    bool is_candidate = false;

    if (vec_idx < num_vectors_in_cluster) {
      // vec load for short factors
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      // Compute inner product using LUT
      float ip = 0.0f;

      // Process each uint32_t of the short code
      for (uint32_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
        // Access short code in transposed layout
        // For transposed layout: vec1[dim0-31], vec2[dim0-31], ..., vecn[dim0-31]
        size_t short_code_offset =
          cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
        uint32_t short_code_chunk = d_short_data[short_code_offset];

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
          ip += __half2float(shared_lut_bf16[lut_offset]);
        }
      }

      // Compute estimated distance
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

      // Compute lower bound
      float low_dist = est_dist - f_error * q_g_error;

      // Check threshold
      //            constexpr float threshold_factor = 1.05;
      //            if (1) {
      if (low_dist < threshold) {
        is_candidate = true;
//                local_low_dist = est_dist;
#ifdef DEBUG_BATCH_SEARCH
//                if (local_low_dist < 0) {
//                    printf ("local_low_dist = %f < 0!\n", local_low_dist);
//                }
#endif
        local_ip = ip;
      }

#ifdef DEBUG_BATCH_SEARCH
      //            if ( threadIdx.x == 0 ) {
//                printf("low_dist: %f, threshold: %f\n", low_dist, threshold);
//                if (low_dist > 1000) {
//                    printf("f_add: %f, q_g_add: %f, f_rescale: %f, ip: %f, q_k1xsumq: %f,
//                    est_dist: %f\n",f_add, q_g_add, f_rescale, ip, q_k1xsumq, est_dist);
//                }
//            }
#endif
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
#ifdef DEBUG_BATCH_SEARCH
      if (threadIdx.x == 10) {
        //                printf("num_candidates: %d\n", num_candidates);
      }
#endif
      if (candidate_slot < max_candidates_per_pair) {
        //                shared_candidate_dists[candidate_slot] = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("1bit estimated distance computation finished!\n");
  }
#endif

  // Step 2 Part 2: Determine which candidates to use
//    int final_num_candidates = min(num_candidates, (int)max_candidates_per_pair);
#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("final_num_candidates_before: %d\n", num_candidates);
  }
#endif

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Sorting TOPK*M finished!\n"); }
#endif

  // Step 3 opt: Compute more accurate distances and select top-k
  // Opt: warp-level dist and then thread-level ex dist restore

  __syncthreads();
  //    __shared__ int probe_slot;
  if (num_candidates > 0) {
    __shared__ int probe_slot;
    {
      using block_sort_t = typename raft::neighbors::ivf_flat::detail::
        flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
      block_sort_t queue(topk);

      // Additional shared values needed for Step 3
      __shared__ float q_kbxsumq;
      if (tid == 0) { q_kbxsumq = d_G_kbxSumq[query_idx]; }
      __syncthreads();

      // Calculate long code parameters
      const uint32_t long_code_size = (D * ex_bits + 7) / 8;

      // Load query vector to shared memory
      float* shared_query = (float*)(shared_buffer);
      for (uint32_t i = tid; i < D; i += num_threads) {
        shared_query[i] = d_query[query_idx * D + i];
      }
      __syncthreads();

      // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

      // Reuse shared_candidate_dists to store IP2 results
      float* shared_ip2_results = reinterpret_cast<float*>(shared_lut_bf16);

      const int warp_id   = tid / WARP_SIZE;
      const int lane_id   = tid % WARP_SIZE;
      const int num_warps = num_threads / WARP_SIZE;

      // Each warp processes different candidates
      for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
        //            int local_vec_idx = ;
        size_t global_vec_idx = cluster_start_index + shared_candidate_indices[cand_idx];

        // Pointer to this vector's long code
        const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;

        // Warp-level IP2 computation
        float ip2 = 0.0f;

        // Each thread in warp processes different dimensions
        for (uint32_t d = lane_id; d < D; d += WARP_SIZE) {
          // Extract ex_bits value for this dimension
          uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
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
#ifdef DEBUG_BATCH_SEARCH
      if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part1 finished!\n"); }
#endif

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
#ifdef DEBUG_BATCH_SEARCH
          if (local_vec_idx > num_vectors_in_cluster) {
            printf("Error! local_vec_index %d moare than num_vectors %d in cluster!\n",
                   local_vec_idx,
                   num_vectors_in_cluster);
          }
#endif

          // vec load version
          float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
//                ex_dist = ex_dist+1;
#ifdef DEBUG_BATCH_SEARCH
          if (ex_dist < 0 && cand_idx < num_candidates) {
            printf("f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f， pos %d in cluster %d\n",
                   f_ex_add,
                   f_ex_rescale,
                   ip,
                   ip2,
                   local_vec_idx,
                   cluster_idx);
            if (cand_idx + 1 < num_candidates) {
              printf("next_data's f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f\n",
                     d_ex_factor[global_vec_idx * 2 + 2],
                     d_ex_factor[global_vec_idx * 2 + 3],
                     shared_candidate_ips[cand_idx + 1],
                     shared_ip2_results[cand_idx + 1]);
            }
          }
#endif
          // Get PID
          pid = (uint32_t)d_pids[global_vec_idx];

        } else {
          // Thread has no valid candidate for this round - use dummy values
          ex_dist = INFINITY;
          pid     = 0;
        }
#ifdef DEBUG_BATCH_SEARCH
        if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
          printf(
            "Wrong pid/ex_dist! PID: %d, ex_dist: %f, ip2: %f, query_idx: %d, max_candidate_num: "
            "%ld, num_cluster_vectors: %ld, cluster idx: %d\n",
            pid,
            ex_dist,
            shared_ip2_results[cand_idx],
            query_idx,
            max_candidates_per_pair,
            num_vectors_in_cluster,
            cluster_idx);
          ex_dist = INFINITY;
        }
#endif
        // ALL threads call queue.add() exactly once per round
        queue.add(ex_dist, pid);
      }

      __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
      if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part2 finished!\n"); }
#endif

      // Step 3 Part 3: Merge results and write back top-k

      queue.done((uint8_t*)shared_lut_bf16);

      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= nprobe) {
        //            printf("Impossible!!!!!!!\n");
        return;
      }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));
    }

    //--------
#ifdef DEBUG_BATCH_SEARCH
    uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
    if (blockIdx.x < 10 && threadIdx.x == 0) {
      printf(
        "dist, pid: %f, %d\n", *(d_topk_dists + output_offset), *(d_topk_pids + output_offset));
      printf("followed dist, pid: %f, %d\n",
             *(d_topk_dists + output_offset + 1),
             *(d_topk_pids + output_offset + 1));
    }
#endif

#ifdef DEBUG_BATCH_SEARCH
    if (threadIdx.x == 0) {
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      if (/*num_candidates < topk ||*/ d_topk_dists[output_offset] < 0) {
        printf("Num candidates = %d < topk = %d \n", num_candidates, topk);
        for (int i = 0; i < topk; i++) {
          printf("pair %d: dist = %f, pid = %d\n",
                 i,
                 *(d_topk_dists + output_offset + i),
                 *(d_topk_pids + output_offset + i));
        }
      }
    }
#endif
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part3 finished!\n"); }
#endif

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Final topk for the cluster get!\n"); }
#endif

    float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      uint32_t output_offset = query_idx * (topk * nprobe) +
                               probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (uint32_t i = 0; i < topk; i++) {
        float dist = d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
      }
    }

    __syncthreads();

    // Update threshold using atomicMin (for floats)
    // max_topk_dist should be > 0 to prevent using initialized memory
    if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
      // Use integer interpretation for atomic operations
      int* threshold_ptr = (int*)(d_threshold + query_idx);
      int new_val        = __float_as_int(max_topk_dist);

      // Atomic minimum for floats (assuming positive distances)
      atomicMin(threshold_ptr, new_val);

#ifdef DEBUG_BATCH_SEARCH
      //        if ( threadIdx.x == 0 ) {
//            printf("Update threshold from %f to %f!\n", threshold, max_topk_dist);
//        }
#endif
      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("TOPK threshold updated!\n"); }
#endif
  }
}

// always compute LUT, designed for first-round nearest cluster search
__global__ void computeInnerProductsWithAlwaysLUT(
  const ClusterQueryPair* d_sorted_pairs,
  const float* d_query,
  const uint32_t* d_short_data,
  const IVFGPU::GPUClusterMeta* d_cluster_meta,
  float* d_lut_for_queries,
  const float* d_short_factors,       // NEW
  const float* d_G_k1xSumq,           // NEW
  const float* d_G_kbxSumq,           // NEW (not used yet)
  const float* d_centroid_distances,  // NEW
                                      //        float* d_distances,  // output distances
  size_t topk,
  size_t num_queries,
  size_t nprobe,
  size_t num_pairs,
  size_t num_centroids,
  size_t D,
  const float* d_threshold,  // NEW: threshold for each query
  //        float* d_ip_results,                 // NEW: store inner products for candidates
  size_t M,                        // NEW: multiplier for topk
  size_t max_candidates_per_pair,  // NEW: max storage per pair, 1000 suggested
  size_t ex_bits,                  // NEW: bits per dimension in ex codes
  const uint8_t* d_long_code,      // NEW: long codes for all vectors
  const float* d_ex_factor,        // NEW: ex factors for distance computation
  const PID* d_pids,               // NEW: PIDs for all vectors
  float* d_topk_dists,             // NEW: output top-k distances
  PID* d_topk_pids,                // NEW: output top-k PIDs
  int* d_query_write_counters)
{
  // Each block handles one <cluster, query> pair

  const int block_id = blockIdx.x;  // simply use 1-D block

  if (block_id >= num_pairs) return;

  // Get the cluster-query pair for this block
  ClusterQueryPair pair = d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  // Check bounds
  if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

  // Get cluster metadata
  size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = d_cluster_meta[cluster_idx].start_index;

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Preparation completed!\n"); }
#endif

  // Calculate LUT parameters
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Shared memory for LUT
  extern __shared__ __align__(256) float shared_lut[];

  // Thread index within the block
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointer to this query's LUT in global memory
  float* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

  // Pointer to this query's vector
  const float* query_vec = d_query + query_idx * D;
  // ------

  // Step 1: Check if LUT needs to be computed and compute if necessary
  bool need_compute_lut = true;

  // If LUT needs to be computed, compute it
  if (need_compute_lut) {
    // Instead of each thread handling one chunk for all LUT entries,
    // have threads cooperatively handle each LUT entry across chunks
    for (int lut_entry = 0; lut_entry < LUT_SIZE; lut_entry++) {
      for (size_t chunk_idx = tid; chunk_idx < num_chunks; chunk_idx += num_threads) {
        size_t dim_start = chunk_idx * BITS_PER_CHUNK;
        float sum        = 0.0f;

        // Compute sum for this chunk and LUT entry
        for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
          size_t dim = dim_start + bit_idx;
          if (dim < D) {
            if (lut_entry & (1 << (BITS_PER_CHUNK - 1 - bit_idx))) { sum += query_vec[dim]; }
          }
        }

        // Now adjacent threads write to adjacent memory locations!
        size_t lut_offset     = lut_entry * num_chunks + chunk_idx;
        query_lut[lut_offset] = sum;
      }
    }

    // Ensure all threads have finished computing their part of the LUT
    __syncthreads();
  }

  // Then Load LUT into shared memory
  // Each thread loads part of the LUT
  for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut[i] = query_lut[i];
  }

  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("LUT computation & load finished!\n"); }
#endif

  // Step 2 Part 1: Compute distances using LUT && decide candidates

  // Shared values for this <cluster, query> pair
  __shared__ float q_g_add;       // squared distance to centroid
  __shared__ float q_k1xsumq;     // query factor
  __shared__ float q_g_error;     // sqrt(q_g_add)
                                  //    __shared__ float threshold;     // threshold for this query
  __shared__ int num_candidates;  // counter for candidates

  // Load shared query-cluster values
  if (tid == 0) {
    // Get squared distance from query to this cluster's centroid
    q_g_add   = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error = sqrtf(q_g_add);

    // Get query factor
    q_k1xsumq = d_G_k1xSumq[query_idx];
    //        threshold = d_threshold[query_idx];  // NEW: load threshold // first round no need to
    //        load infinity threshold
    num_candidates = 0;  // NEW: initialize counter
  }
  __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
  //    if ( threadIdx.x == 0 ) {
//        printf("query idx: %d, threshold after loading: %f\n", query_idx, threshold);
//    }
#endif

  // Allocate shared memory for candidate storage (after LUT)
  // Assuming extern shared memory is large enough
  float* shared_candidate_dists = shared_lut + (num_chunks * LUT_SIZE);
  float* shared_candidate_ips   = shared_candidate_dists + max_candidates_per_pair;
  int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);
  int* shared_buffer            = shared_candidate_indices + max_candidates_per_pair;

  // Calculate short code parameters
  const size_t short_code_length = D / 32;               // number of uint32_t per vector
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
      // vec load for short factors
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
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
        uint32_t short_code_chunk = d_short_data[short_code_offset];

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
      //            constexpr float threshold_factor = 1.05;
      if (1) {  // first round, always threshold
                //            if (low_dist < threshold) {
        is_candidate   = true;
        local_low_dist = est_dist;
#ifdef DEBUG_BATCH_SEARCH
        if (local_low_dist < 0) { printf("local_low_dist = %f < 0!\n", local_low_dist); }
#endif
        local_ip = ip;
      }

#ifdef DEBUG_BATCH_SEARCH
      //            if ( threadIdx.x == 0 ) {
//                printf("low_dist: %f, threshold: %f\n", low_dist, threshold);
//                if (low_dist > 1000) {
//                    printf("f_add: %f, q_g_add: %f, f_rescale: %f, ip: %f, q_k1xsumq: %f,
//                    est_dist: %f\n",f_add, q_g_add, f_rescale, ip, q_k1xsumq, est_dist);
//                }
//            }
#endif
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
#ifdef DEBUG_BATCH_SEARCH
      if (threadIdx.x == 10) {
        //                printf("num_candidates: %d\n", num_candidates);
      }
#endif
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_dists[candidate_slot]   = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("1bit estimated distance computation finished!\n");
  }
#endif

  // Step 2 Part 2: Determine which candidates to use
  // Changed the input parameters so that num_candidates always less or equal than
  // (int)max_candidates_per_pair
  int final_num_candidates = num_candidates;

  // Step 3 opt: Compute more accurate distances and select top-k
  // Opt: warp-level dist and then thread-level ex dist restore

  __syncthreads();
  if (final_num_candidates > 0) {
    using block_sort_t = typename raft::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
    block_sort_t queue(topk);

    // Additional shared values needed for Step 3
    __shared__ float q_kbxsumq;
    if (tid == 0) { q_kbxsumq = d_G_kbxSumq[query_idx]; }
    __syncthreads();

    // Calculate long code parameters
    const size_t long_code_size = (D * ex_bits + 7) / 8;

    // Load query vector to shared memory
    float* shared_query = (float*)(shared_lut);
    for (size_t i = tid; i < D; i += num_threads) {
      shared_query[i] = d_query[query_idx * D + i];
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
      const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;

      // Warp-level IP2 computation
      float ip2 = 0.0f;

      // Each thread in warp processes different dimensions
      for (size_t d = lane_id; d < D; d += WARP_SIZE) {
        // Extract ex_bits value for this dimension
        uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
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
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part1 finished!\n"); }
#endif

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
#ifdef DEBUG_BATCH_SEARCH
        if (local_vec_idx > num_vectors_in_cluster) {
          printf("Error! local_vec_index %d moare than num_vectors %d in cluster!\n",
                 local_vec_idx,
                 num_vectors_in_cluster);
        }
#endif

        // vec load version for ex factors
        float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
        float f_ex_add     = ex_factors.x;
        float f_ex_rescale = ex_factors.y;

        // Compute final distance using pre-computed ip2
        ex_dist = f_ex_add + q_g_add +
                  f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
#ifdef DEBUG_BATCH_SEARCH
        if (ex_dist < 0 && cand_idx < final_num_candidates) {
          printf("f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f， pos %d in cluster %d\n",
                 f_ex_add,
                 f_ex_rescale,
                 ip,
                 ip2,
                 local_vec_idx,
                 cluster_idx);
          if (cand_idx + 1 < final_num_candidates) {
            printf("next_data's f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f\n",
                   d_ex_factor[global_vec_idx * 2 + 2],
                   d_ex_factor[global_vec_idx * 2 + 3],
                   shared_candidate_ips[cand_idx + 1],
                   shared_ip2_results[cand_idx + 1]);
          }
        }
#endif
        // Get PID
        pid = (uint32_t)d_pids[global_vec_idx];

      } else {
        // Thread has no valid candidate for this round - use dummy values
        ex_dist = INFINITY;
        pid     = 0;
      }
#ifdef DEBUG_BATCH_SEARCH
      if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
        printf(
          "Wrong pid/ex_dist! PID: %d, ex_dist: %f, ip2: %f, query_idx: %d, max_candidate_num: "
          "%ld, num_cluster_vectors: %ld\n",
          pid,
          ex_dist,
          shared_candidate_dists[cand_idx],
          query_idx,
          max_candidates_per_pair,
          num_vectors_in_cluster);
        //                ex_dist = INFINITY;
      }
#endif
      // ALL threads call queue.add() exactly once per round
      queue.add(ex_dist, pid);
    }

    __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part2 finished!\n"); }
#endif

    // Step 3 Part 3: Merge results and write back top-k

    uint8_t* queue_buffer = (uint8_t*)shared_buffer;
    queue.done(queue_buffer);

    // Atomically get write position
    __shared__ int probe_slot;
    if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
    __syncthreads();

    if (probe_slot >= nprobe) {
      //            printf("Impossible!!!!!!!\n");
      return;
    }

    // Calculate output offset and store results
    size_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
    queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));

    //--------
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) {
      printf(
        "dist, pid: %f, %d\n", *(d_topk_dists + output_offset), *(d_topk_pids + output_offset));
      printf("followed dist, pid: %f, %d\n",
             *(d_topk_dists + output_offset + 1),
             *(d_topk_pids + output_offset + 1));
    }
#endif

#ifdef DEBUG_BATCH_SEARCH
    if (threadIdx.x == 0) {
      if (/*num_candidates < topk ||*/ d_topk_dists[output_offset] < 0) {
        printf("Num candidates = %d < topk = %d \n", num_candidates, topk);
        for (int i = 0; i < topk; i++) {
          printf("pair %d: dist = %f, pid = %d\n",
                 i,
                 *(d_topk_dists + output_offset + i),
                 *(d_topk_pids + output_offset + i));
        }
      }
    }
#endif
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part3 finished!\n"); }
#endif

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Final topk for the cluster get!\n"); }
#endif

    __shared__ float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      size_t output_offset = query_idx * (topk * nprobe) +
                             probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (size_t i = 0; i < topk; i++) {
        float dist = d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
      }
    }

    __syncthreads();

    // Update threshold using atomicMin (for floats)
    // max_topk_dist should be > 0 to prevent using initialized memory
    // First round, always update if there are valid candidates
    if (tid == 0 && max_topk_dist > 0) {
      // Use integer interpretation for atomic operations
      int* threshold_ptr = (int*)(d_threshold + query_idx);
      int new_val        = __float_as_int(max_topk_dist);

      // Atomic minimum for floats (assuming positive distances)
      atomicMin(threshold_ptr, new_val);

#ifdef DEBUG_BATCH_SEARCH
      //        if ( threadIdx.x == 0 ) {
//            printf("Update threshold from %f to %f!\n", threshold, max_topk_dist);
//        }
#endif
      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("TOPK threshold updated!\n"); }
#endif
  }
}

// assuse LUT always pre-computed; no threshold updates;
__global__ void computeInnerProductsWithLUTWithoutUpdatingThreshold(
  const ClusterQueryPair* d_sorted_pairs,
  const float* d_query,
  const uint32_t* d_short_data,
  const IVFGPU::GPUClusterMeta* d_cluster_meta,
  float* d_lut_for_queries,
  const float* d_short_factors,       // NEW
  const float* d_G_k1xSumq,           // NEW
  const float* d_G_kbxSumq,           // NEW (not used yet)
  const float* d_centroid_distances,  // NEW
                                      //        float* d_distances,  // output distances
  size_t topk,
  size_t num_queries,
  size_t nprobe,
  size_t num_pairs,
  size_t num_centroids,
  size_t D,
  const float* d_threshold,  // NEW: threshold for each query
  //        float* d_ip_results,                 // NEW: store inner products for candidates
  size_t M,                        // NEW: multiplier for topk
  size_t max_candidates_per_pair,  // NEW: max storage per pair, 1000 suggested
  size_t ex_bits,                  // NEW: bits per dimension in ex codes
  const uint8_t* d_long_code,      // NEW: long codes for all vectors
  const float* d_ex_factor,        // NEW: ex factors for distance computation
  const PID* d_pids,               // NEW: PIDs for all vectors
  float* d_topk_dists,             // NEW: output top-k distances
  PID* d_topk_pids,                // NEW: output top-k PIDs
  int* d_query_write_counters)
{
  // Each block handles one <cluster, query> pair
  const int block_id = blockIdx.x;  // simply use 1-D block

  if (block_id >= num_pairs) return;

  // Get the cluster-query pair for this block
  ClusterQueryPair pair = d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  // Check bounds
  if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

  // Get cluster metadata
  //    IVFGPU::GPUClusterMeta cluster_meta = d_cluster_meta[cluster_idx];
  size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = d_cluster_meta[cluster_idx].start_index;

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Preparation completed!\n"); }
#endif

  // Calculate LUT parameters
  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  // Shared memory for LUT
  extern __shared__ __align__(256) float shared_lut[];

  // Thread index within the block
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Pointer to this query's LUT in global memory
  float* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

  // Pointer to this query's vector
  const float* query_vec = d_query + query_idx * D;

  // Then Load LUT into shared memory
  // Each thread loads part of the LUT
  for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut[i] = query_lut[i];
  }

  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) { printf("LUT computation & load finished!\n"); }
#endif

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
    q_g_add   = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error = sqrtf(q_g_add);

    // Get query factor
    q_k1xsumq      = d_G_k1xSumq[query_idx];
    threshold      = d_threshold[query_idx];  // NEW: load threshold
    num_candidates = 0;                       // NEW: initialize counter
  }
  __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
  //    if ( threadIdx.x == 0 ) {
//        printf("query idx: %d, threshold after loading: %f\n", query_idx, threshold);
//    }
#endif

  // Allocate shared memory for candidate storage (after LUT)
  // Assuming extern shared memory is large enough
  float* shared_candidate_dists = shared_lut + (num_chunks * LUT_SIZE);
  float* shared_candidate_ips   = shared_candidate_dists + max_candidates_per_pair;
  int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);
  int* shared_buffer            = shared_candidate_indices + max_candidates_per_pair;

  // Calculate short code parameters
  const size_t short_code_length = D / 32;               // number of uint32_t per vector
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
      // vec load for short factors
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
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
        uint32_t short_code_chunk = d_short_data[short_code_offset];

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
      //            constexpr float threshold_factor = 1.05;
      //            if (1) {
      if (low_dist < threshold) {
        is_candidate   = true;
        local_low_dist = est_dist;
#ifdef DEBUG_BATCH_SEARCH
        if (local_low_dist < 0) { printf("local_low_dist = %f < 0!\n", local_low_dist); }
#endif
        local_ip = ip;
      }

#ifdef DEBUG_BATCH_SEARCH
      //            if ( threadIdx.x == 0 ) {
//                printf("low_dist: %f, threshold: %f\n", low_dist, threshold);
//                if (low_dist > 1000) {
//                    printf("f_add: %f, q_g_add: %f, f_rescale: %f, ip: %f, q_k1xsumq: %f,
//                    est_dist: %f\n",f_add, q_g_add, f_rescale, ip, q_k1xsumq, est_dist);
//                }
//            }
#endif
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
#ifdef DEBUG_BATCH_SEARCH
      if (threadIdx.x == 10) {
        //                printf("num_candidates: %d\n", num_candidates);
      }
#endif
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_dists[candidate_slot]   = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("1bit estimated distance computation finished!\n");
  }
#endif

  // Step 2 Part 2: Determine which candidates to use
  int final_num_candidates = num_candidates;

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("final_num_candidates_before: %d\n", final_num_candidates);
  }
#endif

  // Step 3 opt: Compute more accurate distances and select top-k
  // Opt: warp-level dist and then thread-level ex dist restore

  __syncthreads();
  //    __shared__ int probe_slot;
  if (final_num_candidates > 0) {
    using block_sort_t = typename raft::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
    block_sort_t queue(topk);

    // Additional shared values needed for Step 3
    __shared__ float q_kbxsumq;
    if (tid == 0) { q_kbxsumq = d_G_kbxSumq[query_idx]; }
    __syncthreads();

    // Calculate long code parameters
    const size_t long_code_size = (D * ex_bits + 7) / 8;

    // Load query vector to shared memory
    float* shared_query = (float*)(shared_lut);
    for (size_t i = tid; i < D; i += num_threads) {
      shared_query[i] = d_query[query_idx * D + i];
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
      const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;

      // Warp-level IP2 computation
      float ip2 = 0.0f;

      // Each thread in warp processes different dimensions
      for (size_t d = lane_id; d < D; d += WARP_SIZE) {
        // Extract ex_bits value for this dimension
        uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
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
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part1 finished!\n"); }
#endif

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
#ifdef DEBUG_BATCH_SEARCH
        if (local_vec_idx > num_vectors_in_cluster) {
          printf("Error! local_vec_index %d moare than num_vectors %d in cluster!\n",
                 local_vec_idx,
                 num_vectors_in_cluster);
        }
#endif

        // Load ex factors for this vector
        // vec load version
        float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
        float f_ex_add     = ex_factors.x;
        float f_ex_rescale = ex_factors.y;

        // Compute final distance using pre-computed ip2
        ex_dist = f_ex_add + q_g_add +
                  f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
//                ex_dist = ex_dist+1;
#ifdef DEBUG_BATCH_SEARCH
        if (ex_dist < 0 && cand_idx < final_num_candidates) {
          printf("f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f， pos %d in cluster %d\n",
                 f_ex_add,
                 f_ex_rescale,
                 ip,
                 ip2,
                 local_vec_idx,
                 cluster_idx);
          if (cand_idx + 1 < final_num_candidates) {
            printf("next_data's f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f\n",
                   d_ex_factor[global_vec_idx * 2 + 2],
                   d_ex_factor[global_vec_idx * 2 + 3],
                   shared_candidate_ips[cand_idx + 1],
                   shared_ip2_results[cand_idx + 1]);
          }
        }
#endif
        // Get PID
        pid = (uint32_t)d_pids[global_vec_idx];

      } else {
        // Thread has no valid candidate for this round - use dummy values
        ex_dist = INFINITY;
        pid     = 0;
      }
#ifdef DEBUG_BATCH_SEARCH
      if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
        printf(
          "Wrong pid/ex_dist! PID: %d, ex_dist: %f, ip2: %f, query_idx: %d, max_candidate_num: "
          "%ld, num_cluster_vectors: %d\n",
          pid,
          ex_dist,
          shared_candidate_dists[cand_idx],
          query_idx,
          max_candidates_per_pair,
          num_vectors_in_cluster);
        //                ex_dist = INFINITY;
      }
#endif
      // ALL threads call queue.add() exactly once per round
      queue.add(ex_dist, pid);
    }

    __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part2 finished!\n"); }
#endif

    // Step 3 Part 3: Merge results and write back top-k

    uint8_t* queue_buffer = (uint8_t*)shared_buffer;
    queue.done(queue_buffer);

    // Atomically get write position
    __shared__ int probe_slot;
    if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
    __syncthreads();

    if (probe_slot >= nprobe) {
      //            printf("Impossible!!!!!!!\n");
      return;
    }

    // Calculate output offset and store results
    size_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
    queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));

    //--------

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("TOPK threshold updated!\n"); }
#endif
  }
}

__global__ void computeInnerProductsWithBitwiseOpt(
  const ClusterQueryPair* d_sorted_pairs,
  const float* d_query,
  const uint32_t* d_short_data,  // Transposed bit-packed data
  const IVFGPU::GPUClusterMeta* d_cluster_meta,
  const uint32_t* d_packed_queries,  // Packed query bit planes
  const float* d_widths,             // Query scaling factors
  const float* d_short_factors,
  const float* d_G_k1xSumq,
  const float* d_G_kbxSumq,
  const float* d_centroid_distances,
  uint32_t topk,
  uint32_t num_queries,
  uint32_t nprobe,
  uint32_t num_pairs,
  uint32_t num_centroids,
  uint32_t D,
  const float* d_threshold,
  uint32_t M,
  uint32_t max_candidates_per_pair,
  uint32_t ex_bits,
  const uint8_t* d_long_code,
  const float* d_ex_factor,
  const PID* d_pids,
  float* d_topk_dists,
  PID* d_topk_pids,
  int* d_query_write_counters,
  uint32_t num_bits,  // Added: number of bits (8 for int8)
  uint32_t num_words  // Added: D/32
)
{
  const int block_id = blockIdx.x;
  if (block_id >= num_pairs) return;

  ClusterQueryPair pair = d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

  size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = d_cluster_meta[cluster_idx].start_index;

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr = d_packed_queries + query_idx * num_bits * num_words;
  for (uint32_t i = tid; i < num_bits * num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = d_widths[query_idx]; }
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;
  __shared__ float q_g_error;
  __shared__ float threshold;

  if (tid == 0) {
    q_g_add        = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error      = sqrtf(q_g_add);
    q_k1xsumq      = d_G_k1xSumq[query_idx];
    threshold      = d_threshold[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes =
    max(num_bits * num_words * sizeof(uint32_t), max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + max_candidates_per_pair);
  float* shared_query            = (float*)(shared_candidate_indices + max_candidates_per_pair);
  const size_t short_code_length = D / 32;
  // Step 2 Part 1: Compute bitwise inner products
  const int vectors_per_iteration = num_threads;

  // Optimized first-round IP computation - accumulate on the fly
  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {
    size_t vec_idx = vec_base + tid;

    bool is_candidate        = false;
    float local_ip_quantized = 0;

    if (vec_idx < num_vectors_in_cluster) {
      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      // #pragma unroll 4
      for (int word = 0; word < num_words; ++word) {
        size_t data_offset =
          cluster_start_index * num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = d_short_data[data_offset];

        // Fully unrolled bit processing for better ILP
        accumulator += __popc(shared_packed_query[0 * num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * num_words + word] & data_word) << 2;
        accumulator += __popc(shared_packed_query[3 * num_words + word] & data_word) << 3;
        accumulator += __popc(shared_packed_query[4 * num_words + word] & data_word) << 4;
        accumulator += __popc(shared_packed_query[5 * num_words + word] & data_word) << 5;
        accumulator += __popc(shared_packed_query[6 * num_words + word] & data_word) << 6;
        accumulator -= __popc(shared_packed_query[7 * num_words + word] & data_word) << 7;
      }

      // Restore scale and compute estimated distance
      float ip       = (float)accumulator * query_width;
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
      float low_dist = est_dist - f_error * q_g_error;

      if (low_dist < threshold) {
        is_candidate       = true;
        local_ip_quantized = ip;

#ifdef DEBUG_BATCH_SEARCH
//                local_ip_quantized = est_dist; //debug
//                printf("low distance : %f, local_ip_quantized: %f \n", low_dist,
//                local_ip_quantized);
#endif
      }
    }

    __syncwarp();

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = local_ip_quantized;
        shared_candidate_indices[candidate_slot] = vec_idx;
        // #ifdef DEBUG_BATCH_SEARCH
        //                 printf("Write Successfully!\n");
        // #endif
      }
    }
#ifdef DEBUG_BATCH_SEARCH
//        if (threadIdx.x == 0) {
//            printf("num_vectors in cluster: %d, vec %d finished.\n", num_vectors_in_cluster,
//            vec_idx);
//        }
#endif
  }

  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("1bit estimated distance computation finished!\n");
    printf("final_num_candidates_before: %d\n", num_candidates);
  }
#endif

  if (num_candidates > 0) {
    for (size_t i = tid; i < D; i += num_threads) {
      shared_query[i] = d_query[query_idx * D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < max_candidates_per_pair) {
        int vec_idx = shared_candidate_indices[cand_idx];

        // Compute exact inner product with float query
        float exact_ip = 0.0f;

        // Process each uint32_t of the short code
        for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
          // Access short code in transposed layout
          size_t short_code_offset =
            cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
          uint32_t short_code_chunk = d_short_data[short_code_offset];

          // Process each bit in the uint32_t
          // Note: bit 31 is lowest dimension, bit 0 is highest
#pragma unroll 8
          for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
            size_t dim = uint32_idx * 32 + bit_idx;
            if (dim < D) {
              // Extract bit from MSB to LSB
              int bit_position = 31 - bit_idx;
              bool bit_value   = (short_code_chunk >> bit_position) & 0x1;

              // If bit is 1, add the query value
              if (bit_value) { exact_ip += shared_query[dim]; }
            }
          }
        }

        // Store the exact inner product
#ifdef DEBUG_BATCH_SEARCH
//                printf("Differences between 8 bit ip and full ip %f\n",
//                shared_candidate_ips[cand_idx] - exact_ip);
#endif
        shared_candidate_ips[cand_idx] = exact_ip;
      }
    }

    __syncthreads();
    //    ------------------

    __shared__ int probe_slot;
    {
      using block_sort_t = typename raft::neighbors::ivf_flat::detail::
        flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
      block_sort_t queue(topk);

      // Additional shared values needed for Step 3
      __shared__ float q_kbxsumq;
      if (tid == 0) { q_kbxsumq = d_G_kbxSumq[query_idx]; }
      __syncthreads();

      // Calculate long code parameters
      const uint32_t long_code_size = (D * ex_bits + 7) / 8;

      //            // Load query vector to shared memory (disable when choose to compute exact ip
      //            // for candidates
      //            for (uint32_t i = tid; i < D; i += num_threads) {
      //                shared_query[i] = d_query[query_idx * D + i];
      //            }
      //            __syncthreads();

      // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

      // Reuse shared_candidate_dists to store IP2 results
      float* shared_ip2_results = reinterpret_cast<float*>(shared_mem_raw_2);

      const int warp_id   = tid / WARP_SIZE;
      const int lane_id   = tid % WARP_SIZE;
      const int num_warps = num_threads / WARP_SIZE;

      // Each warp processes different candidates
      for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
        //            int local_vec_idx = ;
        size_t global_vec_idx = cluster_start_index + shared_candidate_indices[cand_idx];

        // Pointer to this vector's long code
        const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;

        // Warp-level IP2 computation
        float ip2 = 0.0f;

        // Each thread in warp processes different dimensions
        for (uint32_t d = lane_id; d < D; d += WARP_SIZE) {
          // Extract ex_bits value for this dimension
          uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
          float ex_val      = (float)code_val;
          ip2 += shared_query[d] * ex_val;
        }

        // Warp-level reduction for ip2
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
        }

        // Lane 0 stores the result
        if (lane_id == 0) {
#ifdef DEBUG_BATCH_SEARCH
//                    if (1) {
//                        printf("ip2: %f\n", ip2);
//                    }
#endif
          shared_ip2_results[cand_idx] = ip2;
        }
      }

      __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
      if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part1 finished!\n"); }
#endif

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
#ifdef DEBUG_BATCH_SEARCH
          if (local_vec_idx > num_vectors_in_cluster) {
            printf("Error! local_vec_index %d moare than num_vectors %d in cluster!\n",
                   local_vec_idx,
                   num_vectors_in_cluster);
          }
#endif

          // vec load version
          float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
//
#ifdef DEBUG_BATCH_SEARCH
          ex_dist = ex_dist + 10000;
          //                    if (ip2 < 0) {
          //                        ex_dist = INFINITY;
          //                    }
          if (ex_dist < 0 && cand_idx < num_candidates) {
            printf("f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f， pos %d in cluster %d\n",
                   f_ex_add,
                   f_ex_rescale,
                   ip,
                   ip2,
                   local_vec_idx,
                   cluster_idx);
            //                    if (cand_idx + 1 < num_candidates) {
            //                        printf("next_data's f_ex_add: %f, f_ex_rescale: %f, ip:%f,
            //                        ip2: %f\n",
            //                               d_ex_factor[global_vec_idx * 2 + 2],
            //                               d_ex_factor[global_vec_idx * 2 + 3],
            //                               shared_candidate_ips[cand_idx + 1],
            //                               shared_ip2_results[cand_idx + 1]);
            //                    }
          }
#endif
          // Get PID
          pid = (uint32_t)d_pids[global_vec_idx];

        } else {
          // Thread has no valid candidate for this round - use dummy values
          ex_dist = INFINITY;
          pid     = 0;
        }
#ifdef DEBUG_BATCH_SEARCH
        __syncthreads();
        if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
          printf(
            "Wrong pid/ex_dist! PID: %d, ex_dist: %f, ip2: %f, query_idx: %d, max_candidate_num: "
            "%ld, num_cluster_vectors: %ld, cluster idx: %d\n",
            pid,
            ex_dist,
            shared_ip2_results[cand_idx],
            query_idx,
            max_candidates_per_pair,
            num_vectors_in_cluster,
            cluster_idx);
          //                ex_dist = INFINITY;
        }
#endif
        // ALL threads call queue.add() exactly once per round
        queue.add(ex_dist, pid);
      }

      __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
      if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part2 finished!\n"); }
#endif

      // Step 3 Part 3: Merge results and write back top-k

      queue.done((uint8_t*)shared_mem_raw_2);

      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= nprobe) {
        //            printf("Impossible!!!!!!!\n");
        return;
      }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));
    }

    //--------
#ifdef DEBUG_BATCH_SEARCH
//        uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
//        if (blockIdx.x < 10 && threadIdx.x == 0) {
//            printf("dist, pid: %f, %d\n", *(d_topk_dists + output_offset), *(d_topk_pids +
//            output_offset)); printf("followed dist, pid: %f, %d\n", *(d_topk_dists + output_offset
//            + 1),
//                   *(d_topk_pids + output_offset + 1));
//        }
#endif

#ifdef DEBUG_BATCH_SEARCH
    if (threadIdx.x == 0) {
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      if (/*num_candidates < topk ||*/ d_topk_dists[output_offset] < 0) {
        printf("Num candidates = %d < topk = %d \n", num_candidates, topk);
        for (int i = 0; i < topk; i++) {
          printf("pair %d: dist = %f, pid = %d\n",
                 i,
                 *(d_topk_dists + output_offset + i),
                 *(d_topk_pids + output_offset + i));
        }
      }
    }
#endif
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part3 finished!\n"); }
#endif

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Final topk for the cluster get!\n"); }
#endif

    float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      uint32_t output_offset = query_idx * (topk * nprobe) +
                               probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (uint32_t i = 0; i < topk; i++) {
        float dist = d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
      }
    }

    __syncthreads();

    // Update threshold using atomicMin (for floats)
    // max_topk_dist should be > 0 to prevent using initialized memory
    if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
      // Use integer interpretation for atomic operations
      int* threshold_ptr = (int*)(d_threshold + query_idx);
      int new_val        = __float_as_int(max_topk_dist);

      // Atomic minimum for floats (assuming positive distances)
      atomicMin(threshold_ptr, new_val);

#ifdef DEBUG_BATCH_SEARCH
      //        if ( threadIdx.x == 0 ) {
//            printf("Update threshold from %f to %f!\n", threshold, max_topk_dist);
//        }
#endif
      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("TOPK threshold updated!\n"); }
#endif
  }
}

// Kernel to init invalid distances
__global__ void initDistancesKernel(float* d_input_dists, size_t total_elements)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total_elements) { d_input_dists[tid] = INFINITY; }
}

__global__ void computeInnerProductsWithBitwiseOpt4bit(
  const ClusterQueryPair* d_sorted_pairs,
  const float* d_query,
  const uint32_t* d_short_data,  // Transposed bit-packed data
  const IVFGPU::GPUClusterMeta* d_cluster_meta,
  const uint32_t* d_packed_queries,  // Packed query bit planes
  const float* d_widths,             // Query scaling factors
  const float* d_short_factors,
  const float* d_G_k1xSumq,
  const float* d_G_kbxSumq,
  const float* d_centroid_distances,
  uint32_t topk,
  uint32_t num_queries,
  uint32_t nprobe,
  uint32_t num_pairs,
  uint32_t num_centroids,
  uint32_t D,
  const float* d_threshold,
  uint32_t M,
  uint32_t max_candidates_per_pair,
  uint32_t ex_bits,
  const uint8_t* d_long_code,
  const float* d_ex_factor,
  const PID* d_pids,
  float* d_topk_dists,
  PID* d_topk_pids,
  int* d_query_write_counters,
  uint32_t num_bits,  // Added: number of bits (8 for int8)
  uint32_t num_words  // Added: D/32
)
{
  const int block_id = blockIdx.x;
  if (block_id >= num_pairs) return;

  ClusterQueryPair pair = d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

  size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = d_cluster_meta[cluster_idx].start_index;

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  // Load packed query bit planes into shared memory
  uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Load this query's packed bit planes
  const uint32_t* query_packed_ptr = d_packed_queries + query_idx * num_bits * num_words;
  for (uint32_t i = tid; i < num_bits * num_words; i += num_threads) {
    shared_packed_query[i] = query_packed_ptr[i];
  }

  // Load query width
  __shared__ float query_width;
  if (tid == 0) { query_width = d_widths[query_idx]; }
  //    float query_width = d_widths[query_idx];
  __syncthreads();

  // Shared values for this <cluster, query> pair
  __shared__ int num_candidates;
  __shared__ float q_g_add;
  __shared__ float q_k1xsumq;
  __shared__ float q_g_error;
  __shared__ float threshold;

  if (tid == 0) {
    q_g_add        = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error      = sqrtf(q_g_add);
    q_k1xsumq      = d_G_k1xSumq[query_idx];
    threshold      = d_threshold[query_idx];
    num_candidates = 0;
  }
  __syncthreads();

  // Allocate shared memory for candidates
  size_t packed_query_bytes =
    max(num_bits * num_words * sizeof(uint32_t), max_candidates_per_pair * sizeof(float));
  float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
  int* shared_candidate_indices =
    reinterpret_cast<int*>(shared_candidate_ips + max_candidates_per_pair);
  float* shared_query            = (float*)(shared_candidate_indices + max_candidates_per_pair);
  const size_t short_code_length = D / 32;
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
      float3 factors       = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      float f_error        = factors.z;

      int32_t accumulator = 0;  // Single accumulator, no array needed

      // Load data once, accumulate directly
      int32_t accumulator2 = 0;
      // #pragma unroll 4
      for (int word = 0; word < num_words; ++word) {
        size_t data_offset =
          cluster_start_index * num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = d_short_data[data_offset];
        accumulator2 += __popc(data_word);
        //                uint32_t data_word = __ldg(d_short_data + data_offset);

        accumulator += __popc(shared_packed_query[0 * num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * num_words + word] & data_word) << 2;
        accumulator -= __popc(shared_packed_query[3 * num_words + word] & data_word)
                       << 3;  // Sign bit
      }

      // Restore scale and compute estimated distance
      //            const float query_error_factor_4bit = 0.5;
      //            float ip = ((float) accumulator + 0.5f * accumulator2) * query_width;
      float ip       = (float)accumulator * query_width;
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
      float low_dist = est_dist - f_error * q_g_error;

      if (low_dist < threshold) {
        is_candidate       = true;
        local_ip_quantized = ip;

#ifdef DEBUG_BATCH_SEARCH
//                local_ip_quantized = est_dist; //debug
//                printf("low distance : %f, local_ip_quantized: %f \n", low_dist,
//                local_ip_quantized);
#endif
      }
    }

    __syncwarp();

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = local_ip_quantized;
        shared_candidate_indices[candidate_slot] = vec_idx;
        // #ifdef DEBUG_BATCH_SEARCH
        //                 printf("Write Successfully!\n");
        // #endif
      }
    }
#ifdef DEBUG_BATCH_SEARCH
//        if (threadIdx.x == 0) {
//            printf("num_vectors in cluster: %d, vec %d finished.\n", num_vectors_in_cluster,
//            vec_idx);
//        }
#endif
  }
  // -----------------

  __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("1bit estimated distance computation finished!\n");
    printf("final_num_candidates_before: %d\n", num_candidates);
  }
#endif

  if (num_candidates > 0) {
    for (size_t i = tid; i < D; i += num_threads) {
      shared_query[i] = d_query[query_idx * D + i];
    }
    __syncthreads();

    //    --------------
    // Step 2 （optional): Load float query and compute exact IPs for candidates
    // Now we can overwrite the packed query with the float query

    // Compute exact float inner products for all candidates
    const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

    for (int c = 0; c < candidates_per_thread; ++c) {
      int cand_idx = tid + c * num_threads;

      if (cand_idx < num_candidates && cand_idx < max_candidates_per_pair) {
        int vec_idx = shared_candidate_indices[cand_idx];

        // Compute exact inner product with float query
        float exact_ip = 0.0f;

        // Process each uint32_t of the short code
        for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
          // Access short code in transposed layout
          size_t short_code_offset =
            cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
          uint32_t short_code_chunk = d_short_data[short_code_offset];

          // Process each bit in the uint32_t
          // Note: bit 31 is lowest dimension, bit 0 is highest
#pragma unroll 8
          for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
            size_t dim = uint32_idx * 32 + bit_idx;
            if (dim < D) {
              // Extract bit from MSB to LSB
              int bit_position = 31 - bit_idx;
              bool bit_value   = (short_code_chunk >> bit_position) & 0x1;

              // If bit is 1, add the query value
              if (bit_value) { exact_ip += shared_query[dim]; }
            }
          }
        }

        // Store the exact inner product
#ifdef DEBUG_BATCH_SEARCH
//                printf("Differences between 8 bit ip and full ip %f\n",
//                shared_candidate_ips[cand_idx] - exact_ip);
#endif
        shared_candidate_ips[cand_idx] = exact_ip;
      }
    }

    __syncthreads();
    //    ------------------

    __shared__ int probe_slot;
    {
      using block_sort_t = typename raft::neighbors::ivf_flat::detail::
        flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
      block_sort_t queue(topk);

      // Additional shared values needed for Step 3
      __shared__ float q_kbxsumq;
      if (tid == 0) { q_kbxsumq = d_G_kbxSumq[query_idx]; }
      __syncthreads();

      // Calculate long code parameters
      const uint32_t long_code_size = (D * ex_bits + 7) / 8;

      //            // Load query vector to shared memory (disable when choose to compute exact ip
      //            // for candidates
      //            for (uint32_t i = tid; i < D; i += num_threads) {
      //                shared_query[i] = d_query[query_idx * D + i];
      //            }
      //            __syncthreads();

      // Step 3 Part 1: Warp-level IP2 computation for better memory coalescing

      // Reuse shared_candidate_dists to store IP2 results
      float* shared_ip2_results = reinterpret_cast<float*>(shared_mem_raw_2);

      const int warp_id   = tid / WARP_SIZE;
      const int lane_id   = tid % WARP_SIZE;
      const int num_warps = num_threads / WARP_SIZE;

      // Each warp processes different candidates
      for (int cand_idx = warp_id; cand_idx < num_candidates; cand_idx += num_warps) {
        //            int local_vec_idx = ;
        size_t global_vec_idx = cluster_start_index + shared_candidate_indices[cand_idx];

        // Pointer to this vector's long code
        const uint8_t* vec_long_code = d_long_code + global_vec_idx * long_code_size;

        // Warp-level IP2 computation
        float ip2 = 0.0f;

        // Each thread in warp processes different dimensions
        for (uint32_t d = lane_id; d < D; d += WARP_SIZE) {
          // Extract ex_bits value for this dimension
          uint32_t code_val = extract_code(vec_long_code, d, ex_bits);
          float ex_val      = (float)code_val;
          ip2 += shared_query[d] * ex_val;
        }

        // Warp-level reduction for ip2
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
          ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
        }

        // Lane 0 stores the result
        if (lane_id == 0) {
#ifdef DEBUG_BATCH_SEARCH
//                    if (1) {
//                        printf("ip2: %f\n", ip2);
//                    }
#endif
          shared_ip2_results[cand_idx] = ip2;
        }
      }

      __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
      if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part1 finished!\n"); }
#endif

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
#ifdef DEBUG_BATCH_SEARCH
          if (local_vec_idx > num_vectors_in_cluster) {
            printf("Error! local_vec_index %d moare than num_vectors %d in cluster!\n",
                   local_vec_idx,
                   num_vectors_in_cluster);
          }
#endif

          // vec load version
          float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);

#ifdef DEBUG_BATCH_SEARCH
          //                    ex_dist = ip;   // debug usage
          //                    ex_dist = ex_dist + 10000;
          //                    if (ip2 < 0) {
          //                        ex_dist = INFINITY;
          //                    }
          if (ex_dist < 0 && cand_idx < num_candidates) {
            printf("f_ex_add: %f, f_ex_rescale: %f, ip:%f, ip2: %f， pos %d in cluster %d\n",
                   f_ex_add,
                   f_ex_rescale,
                   ip,
                   ip2,
                   local_vec_idx,
                   cluster_idx);
            //                    if (cand_idx + 1 < num_candidates) {
            //                        printf("next_data's f_ex_add: %f, f_ex_rescale: %f, ip:%f,
            //                        ip2: %f\n",
            //                               d_ex_factor[global_vec_idx * 2 + 2],
            //                               d_ex_factor[global_vec_idx * 2 + 3],
            //                               shared_candidate_ips[cand_idx + 1],
            //                               shared_ip2_results[cand_idx + 1]);
            //                    }
          }
#endif
          // Get PID
          pid = (uint32_t)d_pids[global_vec_idx];

        } else {
          // Thread has no valid candidate for this round - use dummy values
          ex_dist = INFINITY;
          pid     = 0;
        }
#ifdef DEBUG_BATCH_SEARCH
        __syncthreads();
        if (pid < 0 || pid > 1000000 || ex_dist <= 0) {
          printf(
            "Wrong pid/ex_dist! PID: %d, ex_dist: %f, ip2: %f, query_idx: %d, max_candidate_num: "
            "%ld, num_cluster_vectors: %ld, cluster idx: %d\n",
            pid,
            ex_dist,
            shared_ip2_results[cand_idx],
            query_idx,
            max_candidates_per_pair,
            num_vectors_in_cluster,
            cluster_idx);
          //                ex_dist = INFINITY;
        }
#endif
        // ALL threads call queue.add() exactly once per round
        queue.add(ex_dist, pid);
      }

      __syncthreads();
#ifdef DEBUG_BATCH_SEARCH
      if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part2 finished!\n"); }
#endif

      // Step 3 Part 3: Merge results and write back top-k

      queue.done((uint8_t*)shared_mem_raw_2);

      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= nprobe) {
        //            printf("Impossible!!!!!!!\n");
        return;
      }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));
    }

    //--------
#ifdef DEBUG_BATCH_SEARCH
//        uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
//        if (blockIdx.x < 10 && threadIdx.x == 0) {
//            printf("dist, pid: %f, %d\n", *(d_topk_dists + output_offset), *(d_topk_pids +
//            output_offset)); printf("followed dist, pid: %f, %d\n", *(d_topk_dists + output_offset
//            + 1),
//                   *(d_topk_pids + output_offset + 1));
//        }
#endif

#ifdef DEBUG_BATCH_SEARCH
    if (threadIdx.x == 0) {
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      if (/*num_candidates < topk ||*/ d_topk_dists[output_offset] < 0) {
        printf("Num candidates = %d < topk = %d \n", num_candidates, topk);
        for (int i = 0; i < topk; i++) {
          printf("pair %d: dist = %f, pid = %d\n",
                 i,
                 *(d_topk_dists + output_offset + i),
                 *(d_topk_pids + output_offset + i));
        }
      }
    }
#endif
#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x < 10 && threadIdx.x == 0) { printf("Step3 part3 finished!\n"); }
#endif

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("Final topk for the cluster get!\n"); }
#endif

    float max_topk_dist;

    if (tid == 0) {
      max_topk_dist = -INFINITY;

      // Find the maximum distance in our top-k results
      uint32_t output_offset = query_idx * (topk * nprobe) +
                               probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

      for (uint32_t i = 0; i < topk; i++) {
        float dist = d_topk_dists[output_offset + i];
        if (dist > 0 && dist > max_topk_dist) { max_topk_dist = dist; }
      }
    }

    __syncthreads();

    // Update threshold using atomicMin (for floats)
    // max_topk_dist should be > 0 to prevent using initialized memory
    if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
      // Use integer interpretation for atomic operations
      int* threshold_ptr = (int*)(d_threshold + query_idx);
      int new_val        = __float_as_int(max_topk_dist);

      // Atomic minimum for floats (assuming positive distances)
      atomicMin(threshold_ptr, new_val);

#ifdef DEBUG_BATCH_SEARCH
      //        if ( threadIdx.x == 0 ) {
//            printf("Update threshold from %f to %f!\n", threshold, max_topk_dist);
//        }
#endif
      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0 && threadIdx.x == 0) { printf("TOPK threshold updated!\n"); }
#endif
  }
}

// Kernel to clean distances
__global__ void cleanDistancesKernel(const float* d_input_dists,
                                     float* d_clean_dists,
                                     size_t total_elements)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total_elements) {
    float val = d_input_dists[tid];
    // Replace infinity or NaN with a large valid value
#ifdef DEBUG_BATCH_SEARCH
    if (val < 0) {
      //            printf("Error!! Distance is %f \n", val);
    }
#endif
    if (!isfinite(val) || isnan(val) || val < 0) {
      d_clean_dists[tid] = 1e10f;  // Large but valid distance
    } else {
      d_clean_dists[tid] = val;
    }
  }
}

// Final function to merge top-k results from all clusters
void mergeClusterTopKFinal(const float* d_topk_dists,  // Input: top-k distances from all clusters
                           const PID* d_topk_pids,     // Input: top-k PIDs from all clusters
                           float* d_final_dists,  // Output: final top-k distances for each query
                           PID* d_final_pids,     // Output: final top-k PIDs for each query
                           size_t num_queries,
                           size_t nprobe,
                           size_t topk,
                           bool sorted         = true,  // Whether to sort the final results
                           cudaStream_t stream = 0)
{
  // Create RAFT resources
  raft::device_resources handle(stream);

  size_t candidates_per_query = nprobe * topk;
#ifdef DEBUG_BATCH_SEARCH
  size_t total_elements = num_queries * candidates_per_query;

  // Allocate temporary array for cleaned data
  float* d_clean_dists;
  cudaMalloc(&d_clean_dists, total_elements * sizeof(float));

  //    // Clean the input distances
  //    int threads = 256;
  //    int blocks = (total_elements + threads - 1) / threads;
  //
  //    cleanDistancesKernel<<<blocks, threads, 0, stream>>>(
  //            d_topk_dists,
  //            d_clean_dists,
  //            total_elements
  //    );

  //    cudaStreamSynchronize(stream);  // Ensure cleaning is done
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    cudaFree(d_clean_dists);
    throw std::runtime_error(std::string("Error in cleaning kernel: ") + cudaGetErrorString(err));
  }
#endif

  raft::matrix::detail::select_k(handle,
                                 d_topk_dists,
                                 d_topk_pids,
                                 num_queries,
                                 candidates_per_query,
                                 topk,
                                 d_final_dists,
                                 d_final_pids,
                                 true);
#ifdef DEBUG_BATCH_SEARCH
  std::cout << "Distances merged!" << std::endl;
#endif

//    // Synchronize if needed
//    if (stream == 0) {
//        cudaDeviceSynchronize();
//    } else {
//        cudaStreamSynchronize(stream);
//    }
#ifdef DEBUG_BATCH_SEARCH
  float h_topk_dist;
  PID h_topk_pid;
  cudaMemcpyAsync(&h_topk_dist, d_final_dists, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&h_topk_pid, d_final_pids, sizeof(PID), cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();
  std::cout << h_topk_dist << std::endl;
  std::cout << h_topk_pid << std::endl;

  cudaMemcpyAsync(&h_topk_dist, d_final_dists + 1, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&h_topk_pid, d_final_pids + 1, sizeof(PID), cudaMemcpyDeviceToHost, stream);
  cudaDeviceSynchronize();
  std::cout << h_topk_dist << std::endl;
  std::cout << h_topk_pid << std::endl;

  // Clean up
  cudaFree(d_clean_dists);
#endif
}

#ifdef DEBUG_BATCH_SEARCH
void checkAndPrintNegativeValues(ClusterQueryPair* d_sorted_pairs,
                                 size_t num_queries,
                                 size_t nprobe)
{
  size_t total_pairs = num_queries * nprobe;

  // Allocate host memory
  std::vector<ClusterQueryPair> h_pairs(total_pairs);

  // Copy data from GPU to CPU
  cudaMemcpy(
    h_pairs.data(), d_sorted_pairs, total_pairs * sizeof(ClusterQueryPair), cudaMemcpyDeviceToHost);

  // Check for negative values and print them
  bool found_negative = false;
  int negative_count  = 0;

  std::cout << "Checking for negative values in " << total_pairs << " pairs...\n";

  for (size_t i = 0; i < total_pairs; i++) {
    if (h_pairs[i].cluster_idx < 0 || h_pairs[i].query_idx < 0) {
      found_negative = true;
      negative_count++;

      // Print the abnormal value with its position
      std::cout << "Negative value found at index " << i << " (query=" << i / nprobe
                << ", probe=" << i % nprobe << "): "
                << "cluster_idx=" << h_pairs[i].cluster_idx
                << ", query_idx=" << h_pairs[i].query_idx << "\n";

      // Optional: limit printing if there are too many
      if (negative_count >= 100) {
        std::cout << "... (stopping after 100 negative values)\n";
        break;
      }
    }
  }

  if (!found_negative) {
    std::cout << "No negative values found. All pairs are valid.\n";
  } else {
    // Count total if we stopped early
    if (negative_count >= 100) {
      negative_count = 0;
      for (size_t i = 0; i < total_pairs; i++) {
        if (h_pairs[i].cluster_idx < 0 || h_pairs[i].query_idx < 0) { negative_count++; }
      }
    }
    std::cout << "\nTotal negative/abnormal pairs found: " << negative_count << " out of "
              << total_pairs << " (" << (100.0 * negative_count / total_pairs) << "%)\n";
  }
}
#endif

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
                                          PID* d_final_pids,
                                          cudaStream_t stream)
{
#ifdef DEBUG_BATCH_SEARCH
  // check whether pairs are wrong
  checkAndPrintNegativeValues(d_sorted_pairs, num_queries, nprobe);
#endif
  // First allocate space for LUT
  size_t lut_size =
    num_queries * (cur_ivf.num_padded_dim / BITS_PER_CHUNK) * LUT_SIZE * sizeof(float);
  // each line's space is (cur_ivf.num_padded_dim / BITS_PER_CHUNK) * LUT_SIZE * sizeof(float);
  float* d_lut_for_queries = nullptr;
  cudaMallocAsync(&d_lut_for_queries, lut_size, stream);
  thrust::fill(thrust::cuda::par.on(stream),
               d_lut_for_queries,
               d_lut_for_queries + (lut_size / sizeof(float)),
               -std::numeric_limits<float>::infinity());  // initially set to INVALID value
  // precompute LUTS
  launchPrecomputeLUTs(d_query, d_lut_for_queries, num_queries, D, stream);
  // #ifdef DEBUG_BATCH_SEARCH
  //  Clean the input distances
  size_t candidates_per_query = nprobe * topk;
  size_t total_elements       = num_queries * candidates_per_query;
  int threads                 = 256;
  int blocks                  = (total_elements + threads - 1) / threads;

  initDistancesKernel<<<blocks, threads, 0, stream>>>(d_topk_dists, total_elements);
  // #endif
  int* d_query_write_counters;  // One counter per query, indicates where to store final results
                                // (0~nprobe)
  cudaMallocAsync(&d_query_write_counters, num_queries * sizeof(int), stream);
  cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream);  // Initialize to 0

  float* d_topk_threshold_batch;
  cudaMallocAsync(&d_topk_threshold_batch, sizeof(float) * num_queries, stream);
  thrust::fill(thrust::cuda::par.on(stream),
               d_topk_threshold_batch,
               d_topk_threshold_batch + num_queries,
               std::numeric_limits<float>::infinity());
  // Then launch kernel for computation
  size_t num_pairs = num_queries * nprobe;
  dim3 gridDim(num_pairs, 1, 1);
  dim3 blockDim(256, 1, 1);
  size_t num_chunks = D / BITS_PER_CHUNK;
  size_t candidate_storage =
    cur_ivf.max_cluster_length * (2 * sizeof(float) + sizeof(int));  // ip, idx
  size_t query_storage = D * sizeof(float);                          // For shared query vector
  //    std::cout << "trying to compute distances" << std::endl;
  //    if (MAX_TOP_K < cur_ivf.max_cluster_length) {
  //        throw std::runtime_error(
  //                "MAX_TOP_K (" + std::to_string(MAX_TOP_K) +
  //                ") < max_cluster_length (" + std::to_string(cur_ivf.max_cluster_length) + ")");
  //    }
  const int smem_bytes =
    raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(blockDim.x / 32,
                                                                                   MAX_TOP_K);
  size_t shared_mem_size =
    num_chunks * LUT_SIZE * sizeof(float) + candidate_storage + query_storage + smem_bytes;
//    printf("LUT part: %.2f KB\n", (num_chunks * LUT_SIZE * sizeof(float)) / 1024.0f);
//    printf("Candidate storage: %.2f KB\n", candidate_storage / 1024.0f);
//    printf("Query storage: %.2f KB\n", query_storage / 1024.0f);
//    printf("Other smem: %.2f KB\n", smem_bytes / 1024.0f);
//    printf("Total shared_mem_size: %.2f KB\n", shared_mem_size / 1024.0f);
#ifdef DEBUG_BATCH_SEARCH
  printf("num_chunks: %d, candidate_storage: %d, query_storage: %d, smem_bytes: %d\n",
         num_chunks,
         candidate_storage,
         query_storage,
         smem_bytes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error Before Launching Distance Kernel! ") +
                             cudaGetErrorString(err));
  }
  printf("shared_mem_size in KB: %lu\n", shared_mem_size / 1024);
#endif
  // Note that for large dimensions, we need to set it for specific kernel
  if (shared_mem_size > 49152) {
    // for larger dimensions
#ifdef DEBUG_BATCH_SEARCH
    printf("Using larger shared memory of %d:\n", shared_mem_size);
#endif
    cudaFuncSetAttribute(computeInnerProductsWithLUT,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         98304);  // 96KB for ampere devices
  }
#ifdef DEBUG_BATCH_SEARCH
  printf("ivf.max_cluster_length: %d\n", cur_ivf.max_cluster_length);
#endif
  computeInnerProductsWithLUT<<<gridDim, blockDim, shared_mem_size, stream>>>(
    d_sorted_pairs,
    d_query,
    cur_ivf.d_short_data,
    d_cluster_meta,
    d_lut_for_queries,
    cur_ivf.d_short_factors_batch,
    d_G_k1xSumq,
    d_G_kbxSumq,
    d_centroid_distances,
    topk,
    num_queries,
    nprobe,
    num_pairs,
    cur_ivf.num_centroids,
    D,
    d_topk_threshold_batch,
    15,  // by default just set amplification vector to 10
    cur_ivf.max_cluster_length,
    cur_ivf.ex_bits,
    cur_ivf.d_long_code,
    reinterpret_cast<const float*>(cur_ivf.d_ex_factor),
    cur_ivf.d_ids,
    d_topk_dists,
    d_topk_pids,
    d_query_write_counters);

#ifdef DEBUG_BATCH_SEARCH
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error Before Merge! ") + cudaGetErrorString(err));
  }
#endif

  // merge results from different blocks
  mergeClusterTopKFinal(d_topk_dists,
                        d_topk_pids,
                        d_final_dists,
                        d_final_pids,
                        num_queries,
                        nprobe,
                        topk,
                        true,  // sorted=true for ordered results
                        stream);

  //    std::cout << "block distances merged!" << std::endl;
  cudaFreeAsync(d_topk_threshold_batch, stream);
  cudaFreeAsync(d_lut_for_queries, stream);
  cudaFreeAsync(d_query_write_counters, stream);
}

// Simpler non-optimized version with BF16
__global__ void precomputeAllLUTs_bf16_simple(const float* d_query,
                                              lut_dtype* d_lut_for_queries,
                                              size_t num_queries,
                                              size_t D)
{
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  lut_dtype* query_lut   = d_lut_for_queries + query_idx * lut_per_query_size;
  const float* query_vec = d_query + query_idx * D;

  for (size_t chunk_idx = tid; chunk_idx < num_chunks; chunk_idx += num_threads) {
    size_t dim_start = chunk_idx * BITS_PER_CHUNK;

    for (int lut_entry = 0; lut_entry < LUT_SIZE; lut_entry++) {
      float sum = 0.0f;  // Compute in FP32

      for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
        size_t dim = dim_start + bit_idx;
        if (dim < D) {
          if (lut_entry & (1 << (BITS_PER_CHUNK - 1 - bit_idx))) { sum += query_vec[dim]; }
        }
      }

      size_t lut_offset = chunk_idx * LUT_SIZE + lut_entry;
      // Convert to BF16 when storing
      //            query_lut[lut_offset] = __float2bfloat16(sum);
      query_lut[lut_offset] = __float2half(sum);
    }
  }
}

__global__ void precomputeAllLUTs_bf16_optimized(const float* d_query,
                                                 lut_dtype* d_lut_for_queries,  // Output in BF16
                                                 size_t num_queries,
                                                 size_t D)
{
  const int query_idx = blockIdx.x;
  if (query_idx >= num_queries) return;

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Shared memory for query chunk and temporary LUT storage (still use FP32 for computation)
  extern __shared__ float shared_mem[];
  float* shared_query = shared_mem;
  float* shared_lut   = shared_mem + BITS_PER_CHUNK;

  const size_t num_chunks         = D / BITS_PER_CHUNK;
  const size_t lut_per_query_size = num_chunks * LUT_SIZE;

  lut_dtype* query_lut   = d_lut_for_queries + query_idx * lut_per_query_size;
  const float* query_vec = d_query + query_idx * D;

  for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    size_t dim_start = chunk_idx * BITS_PER_CHUNK;

    // Load query chunk into shared memory
    if (tid < BITS_PER_CHUNK && dim_start + tid < D) {
      shared_query[tid] = query_vec[dim_start + tid];
    }
    __syncthreads();

    // Compute LUT entries in FP32
    for (int lut_entry = tid; lut_entry < LUT_SIZE; lut_entry += num_threads) {
      float sum = 0.0f;

#pragma unroll
      for (int bit_idx = 0; bit_idx < BITS_PER_CHUNK; bit_idx++) {
        if (dim_start + bit_idx < D) {
          if (lut_entry & (1 << (BITS_PER_CHUNK - 1 - bit_idx))) { sum += shared_query[bit_idx]; }
        }
      }

      if (lut_entry < LUT_SIZE) { shared_lut[lut_entry] = sum; }
    }
    __syncthreads();

    // Coalesced write to global memory with BF16 conversion
    size_t base_offset = chunk_idx * LUT_SIZE;
    for (int i = tid; i < LUT_SIZE; i += num_threads) {
      // Convert FP32 to BF16 during store
      //            query_lut[base_offset + i] = __float2bfloat16(shared_lut[i]);
      query_lut[base_offset + i] = __float2half(shared_lut[i]);
    }
    __syncthreads();
  }
}

// Optimized version with BF16
void launchPrecomputeLUTs_bf16(const float* d_query,
                               lut_dtype* d_lut_for_queries,
                               size_t num_queries,
                               size_t D,
                               cudaStream_t stream = 0,
                               bool use_optimized  = false)
{
  dim3 gridDim(num_queries, 1, 1);
  dim3 blockDim(256, 1, 1);

  if (use_optimized) {
    size_t shared_mem_size = (BITS_PER_CHUNK + LUT_SIZE) * sizeof(float);
    precomputeAllLUTs_bf16_optimized<<<gridDim, blockDim, shared_mem_size, stream>>>(
      d_query, d_lut_for_queries, num_queries, D);
  } else {
    precomputeAllLUTs_bf16_simple<<<gridDim, blockDim, 0, stream>>>(
      d_query, d_lut_for_queries, num_queries, D);
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
  //        const int*   __restrict__ d_bin_XP,
  //        const float* __restrict__ d_XP_norm,
  const float* __restrict__ d_XP,
  //        const float* __restrict__ d_centroid,
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
  //    float*   s_xp_norm   = s_mem;
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

  // Normalize queries
  //    if (norm > 0) {
  //        for (int j = tid; j < D; j += BlockSize) {
  //            s_xp_norm[j] = s_xp[j] / norm;  // XP/norm(XP)
  //        }
  //    } else {
  //        for (int j = tid; j < D; j += BlockSize) {
  //            s_xp_norm[j] = 0.0f;
  //        }
  //    }
  //    __syncthreads();

  //=========================================================================
  // Step 1: Coalesced load of all necessary data into shared memory
  //=========================================================================
  //    for (int j = tid; j < D; j += BlockSize) {
  ////        s_xp_norm[j]  = d_XP_norm[row * D + j];
  ////        s_bin_xp[j]   = d_bin_XP[row * D + j];
  //        s_xp[j]       = d_XP[row * D + j];
  //    }
  //    __syncthreads();

  //=========================================================================
  // Part A: ExRaBitQ Code Generation
  //=========================================================================
  //    const int mask = (1 << EX_BITS) - 1;
  //    float thread_ipnorm_sum = 0.0f;

  // Parallel quantization and start of ip_norm reduction
  for (int j = tid; j < D; j += BlockSize) {
    float val    = s_xp[j] * norm_inv;
    int code_val = __float2int_rn((const_scaling_factor * val) /*+ 0.5*/);  // 4 drop and 5 in
    if (code_val > (1 << (EX_BITS - 1)) - 1) code_val = (1 << (EX_BITS - 1)) - 1;
    if (code_val < (-(1 << (EX_BITS - 1)))) code_val = -(1 << (EX_BITS - 1));
    s_tmp_code[j] = code_val;
    //        thread_ipnorm_sum += (code_val) * val;  // remove delta/2
  }
  __syncthreads();

  // Parallel bit-flipping
  //    for (int j = tid; j < D; j += BlockSize) {
  //        if (s_bin_xp[j] == 0) {
  //            s_tmp_code[j] = (~s_tmp_code[j]) & mask;
  //        }
  //    }
  //    __syncthreads();

  // Finish ip_norm reduction
  //    float total_ipnorm = blockReduceSum(thread_ipnorm_sum);
  //    float ip_norm_inv = 1.0f;
  //    if (tid == 0) {
  //        float inv = 1.0f / total_ipnorm;
  //        ip_norm_inv = isfinite(inv) ? inv : 1.0f;
  //    }
  //    // Broadcast ip_norm_inv to all threads in the block
  //    ip_norm_inv = __shfl_sync(0xffffffff, ip_norm_inv, 0);

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
  //    l2_sqr       = blockReduceSum(l2_sqr);
  ip_resi_xucb = blockReduceSum(ip_resi_xucb);
  //    ip_cent_xucb = blockReduceSum(ip_cent_xucb);
  xu_sq = blockReduceSum(xu_sq);

  // Thread 0 computes and writes the final factors
  if (tid == 0) {
    //        float denom = ip_resi_xucb;
    //        if (denom == 0.0f) denom = INFINITY;
    //        float l2_norm = sqrtf(fmaxf(l2_sqr, 0.f));
    float norm_quan      = sqrtf(fmaxf(xu_sq, 0.f));
    float cos_similarity = ip_resi_xucb / (norm * norm_quan);
    //
    //        float fadd     = l2_sqr + 2.f * l2_sqr * ip_cent_xucb / denom;
    //        float frescale = -2.f * l2_norm * ip_norm_inv;
    //
    //
    //        float ratio = (l2_sqr * xu_sq) / (ip_resi_xucb * ip_resi_xucb);
    //        float inner = (ratio - 1.f) / fmaxf(float(D - 1), 1.f);
    //        float tmp_error = l2_norm * kConstEpsilon * sqrtf(fmaxf(inner, 0.f));
    //        float ferr = 2.f * tmp_error;
    float delta = norm / norm_quan * cos_similarity;
    //
    //        float delta = norm / norm_quan;

    size_t base   = row;
    d_delta[base] = delta;
    //        d_ex_factor[base + 1] = frescale;
    //        d_ex_factor[base + 2] = ferr;
  }

  //=========================================================================
  // Part C: Pack and Write Long Code (MINIMAL READS, PARALLEL, COALESCED)
  //=========================================================================
  int long_code_length = D;  // D dims, then D bytes
  int8_t* out_ptr      = d_long_code + row * long_code_length;
  for (int j = tid; j < D; j += BlockSize)
    out_ptr[j] = s_tmp_code[j];  // write outputs directly
}

// optimize shared memory usage using (1) fp16 (2) shared memory reuse in the kernel
void SearcherGPU::SearchClusterQueryPairsSharedMemOpt(
  const IVFGPU& cur_ivf,
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
  PID* d_final_pids,
  cudaStream_t stream)
{
#ifdef DEBUG_BATCH_SEARCH
  // check whether pairs are wrong
  checkAndPrintNegativeValues(d_sorted_pairs, num_queries, nprobe);
#endif
  // Using BF16 for storage

  // Allocate space for LUT with reduced precision
  size_t lut_elements = num_queries * (cur_ivf.num_padded_dim / BITS_PER_CHUNK) * LUT_SIZE;
  size_t lut_size     = lut_elements * sizeof(lut_dtype);

  lut_dtype* d_lut_for_queries = nullptr;
  cudaMallocAsync(&d_lut_for_queries, lut_size, stream);

  // Initialize with -infinity (convert to BF16)
  float neg_inf = -std::numeric_limits<float>::infinity();
  //    lut_dtype neg_inf_bf16 = __float2bfloat16(neg_inf);  // Or __float2half(neg_inf) for FP16
  lut_dtype neg_inf_bf16 = __float2half(neg_inf);
  // Fill using thrust with BF16 value
  thrust::fill(thrust::cuda::par.on(stream),
               d_lut_for_queries,
               d_lut_for_queries + lut_elements,
               neg_inf_bf16);

  // Precompute LUTs
  launchPrecomputeLUTs_bf16(
    d_query, d_lut_for_queries, num_queries, cur_ivf.num_padded_dim, stream);
  // #ifdef DEBUG_BATCH_SEARCH
  //  Clean the input distances
  size_t candidates_per_query = nprobe * topk;
  size_t total_elements       = num_queries * candidates_per_query;
  int threads                 = 256;
  int blocks                  = (total_elements + threads - 1) / threads;

  initDistancesKernel<<<blocks, threads, 0, stream>>>(d_topk_dists, total_elements);
  // #endif
  int* d_query_write_counters;  // One counter per query, indicates where to store final results
                                // (0~nprobe)
  cudaMallocAsync(&d_query_write_counters, num_queries * sizeof(int), stream);
  cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream);  // Initialize to 0

  float* d_topk_threshold_batch;
  cudaMallocAsync(&d_topk_threshold_batch, sizeof(float) * num_queries, stream);
  thrust::fill(thrust::cuda::par.on(stream),
               d_topk_threshold_batch,
               d_topk_threshold_batch + num_queries,
               std::numeric_limits<float>::infinity());
  // Then launch kernel for computation
  size_t num_pairs = num_queries * nprobe;
  dim3 gridDim(num_pairs, 1, 1);
  dim3 blockDim(256, 1, 1);
  size_t num_chunks = D / BITS_PER_CHUNK;
  //    size_t candidate_storage = cur_ivf.max_cluster_length * (2 * sizeof(float) + sizeof(int));
  //    // ip, idx
  size_t query_storage = D * sizeof(float);  // For shared query vector
  const int smem_bytes =
    raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(blockDim.x / 32,
                                                                                   MAX_TOP_K);
  size_t first_part_shared_mem =
    max(lut_size / num_queries, cur_ivf.max_cluster_length * (sizeof(float)));
  size_t second_part_shared_mem = cur_ivf.max_cluster_length * (sizeof(float) + sizeof(int));
  size_t third_part_shared_mem  = query_storage;
  // smem reuses first 3 parts
  size_t shared_mem_size =
    max(first_part_shared_mem + second_part_shared_mem + third_part_shared_mem, (size_t)smem_bytes);
//    printf("Shared memory breakdown:\n");
//    printf("  smem_bytes          : %d bytes (%.2f KB)\n",
//           shared_mem_size, shared_mem_size / 1024.0f);
//    printf("  First part (LUT/max cluster floats): %zu bytes (%.2f KB)\n",
//           first_part_shared_mem, first_part_shared_mem / 1024.0f);
//    printf("  Second part (cluster float+int)    : %zu bytes (%.2f KB)\n",
//           second_part_shared_mem, second_part_shared_mem / 1024.0f);
//    printf("  Third part (query_storage)         : %zu bytes (%.2f KB)\n",
//           third_part_shared_mem, third_part_shared_mem / 1024.0f);
//    printf("Total shared_mem_size: %.2f KB\n", shared_mem_size / 1024.0f);
#ifdef DEBUG_BATCH_SEARCH
  //    printf("num_chunks: %d, candidate_storage: %d, query_storage: %d, smem_bytes: %d\n",
  //    num_chunks, candidate_storage, query_storage, smem_bytes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error Before Launching Distance Kernel! ") +
                             cudaGetErrorString(err));
  }
  printf("shared_mem_size in KB: %lu\n", shared_mem_size / 1024);
#endif
  // Note that for large dimensions, we need to set it for specific kernel
  if (shared_mem_size > 49152) {
    // for larger dimensions
#ifdef DEBUG_BATCH_SEARCH
    printf("Using larger shared memory of %d:\n", shared_mem_size);
#endif
    cudaFuncSetAttribute(computeInnerProductsWithLUT,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         98304);  // 96KB for ampere devices
  }
#ifdef DEBUG_BATCH_SEARCH
  printf("ivf.max_cluster_length: %d\n", cur_ivf.max_cluster_length);
#endif

  computeInnerProductsWithLUT16Opt<<<gridDim, blockDim, shared_mem_size, stream>>>(
    d_sorted_pairs,
    d_query,
    cur_ivf.d_short_data,
    d_cluster_meta,
    d_lut_for_queries,
    cur_ivf.d_short_factors_batch,
    d_G_k1xSumq,
    d_G_kbxSumq,
    d_centroid_distances,
    topk,
    num_queries,
    nprobe,
    num_pairs,
    cur_ivf.num_centroids,
    D,
    d_topk_threshold_batch,
    15,  // by default just set amplification vector to 10
    cur_ivf.max_cluster_length,
    cur_ivf.ex_bits,
    cur_ivf.d_long_code,
    reinterpret_cast<const float*>(cur_ivf.d_ex_factor),
    cur_ivf.d_ids,
    d_topk_dists,
    d_topk_pids,
    d_query_write_counters);

#ifdef DEBUG_BATCH_SEARCH
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error Before Merge! ") + cudaGetErrorString(err));
  }
#endif

  // merge results from different blocks
  mergeClusterTopKFinal(d_topk_dists,
                        d_topk_pids,
                        d_final_dists,
                        d_final_pids,
                        num_queries,
                        nprobe,
                        topk,
                        true,  // sorted=true for ordered results
                        stream);

  //    std::cout << "block distances merged!" << std::endl;
  cudaFreeAsync(d_topk_threshold_batch, stream);
  cudaFreeAsync(d_lut_for_queries, stream);
  cudaFreeAsync(d_query_write_counters, stream);
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

  // jamxia edit
  // float block_min = BlockReduceFloat(temp_storage_min).Reduce(local_min, cub::Min());
  float block_min = BlockReduceFloat(temp_storage_min).Reduce(local_min, cuda::minimum<>{});
  __syncthreads();
  // jamxia edit
  // float block_max = BlockReduceFloat(temp_storage_max).Reduce(local_max, cub::Max());
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

// Replace SearcherGPU::SearchClusterQueryPairsSharedMemOpt with this version
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
  cudaStream_t stream,
  bool use_4bit  // Add parameter to choose 4-bit or 8-bit
)
{
  // query quantize
  const int num_bits  = use_4bit ? 4 : 8;  // Choose bit width
  const int num_words = (cur_ivf.num_padded_dim + 31) / 32;

  // Allocate memory for quantization
  float* d_query_ranges;
  float* d_widths;
  int8_t* d_quantized_queries;
  uint32_t* d_packed_queries;

  size_t ranges_size    = num_queries * 2 * sizeof(float);
  size_t widths_size    = num_queries * sizeof(float);
  size_t quantized_size = num_queries * cur_ivf.num_padded_dim * sizeof(int8_t);
  size_t packed_size    = num_queries * num_bits * num_words * sizeof(uint32_t);

  cudaMallocAsync(&d_query_ranges, ranges_size, stream);
  cudaMallocAsync(&d_widths, widths_size, stream);
  cudaMallocAsync(&d_quantized_queries, quantized_size, stream);
  cudaMallocAsync(&d_packed_queries, packed_size, stream);

  if (rabitq_quantize_flag) {
    const int block_size = 256;
    const int grid_size  = num_queries;
    size_t shared_mem    = D * sizeof(float) + D * sizeof(int8_t) + block_size * sizeof(float);
    exrabitq_quantize_query<block_size>
      <<<grid_size, block_size, shared_mem, stream>>>(d_query,
                                                      num_queries,
                                                      D,
                                                      num_bits,
                                                      best_rescaling_factor,
                                                      1.9f,
                                                      d_quantized_queries,
                                                      d_widths);
  } else {  // scalar quantize
    // Step 1: Find min/max for each query
    {
      dim3 block(256);
      dim3 grid(num_queries);
      findQueryRanges<<<grid, block, 0, stream>>>(
        d_query, d_query_ranges, num_queries, cur_ivf.num_padded_dim);
    }

    // Step 2: Quantize queries to int8_t with BQ=8
    {
      const int block_size = 256;
      //        const int grid_size = (num_queries * cur_ivf.num_padded_dim + block_size - 1) /
      //        block_size;
      const int grid_size = num_queries;
      if (use_4bit) {
        quantizeQueriesToInt4<<<grid_size, block_size, 0, stream>>>(d_query,
                                                                    d_query_ranges,
                                                                    d_quantized_queries,
                                                                    d_widths,
                                                                    num_queries,
                                                                    cur_ivf.num_padded_dim);
      } else {
        quantizeQueriesToInt8<<<grid_size, block_size, 0, stream>>>(d_query,
                                                                    d_query_ranges,
                                                                    d_quantized_queries,
                                                                    d_widths,
                                                                    num_queries,
                                                                    cur_ivf.num_padded_dim);
      }
    }
  }

  // Step 3: Pack quantized queries into bit planes
  {
    const int block_size = 256;
    const int grid_size  = (num_queries * num_bits * num_words + block_size - 1) / block_size;

    if (use_4bit) {
      packInt4QueryBitPlanes<<<grid_size, block_size, 0, stream>>>(
        d_quantized_queries, d_packed_queries, num_queries, cur_ivf.num_padded_dim);
    } else {
      packInt8QueryBitPlanes<<<grid_size, block_size, 0, stream>>>(
        d_quantized_queries, d_packed_queries, num_queries, cur_ivf.num_padded_dim);
    }
  }

  // Initialize distances
  size_t candidates_per_query = nprobe * topk;
  size_t total_elements       = num_queries * candidates_per_query;
  int threads                 = 256;
  int blocks                  = (total_elements + threads - 1) / threads;
  initDistancesKernel<<<blocks, threads, 0, stream>>>(d_topk_dists, total_elements);

  int* d_query_write_counters;
  cudaMallocAsync(&d_query_write_counters, num_queries * sizeof(int), stream);
  cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream);

  float* d_topk_threshold_batch;
  cudaMallocAsync(&d_topk_threshold_batch, sizeof(float) * num_queries, stream);
  thrust::fill(thrust::cuda::par.on(stream),
               d_topk_threshold_batch,
               d_topk_threshold_batch + num_queries,
               std::numeric_limits<float>::infinity());

  // Launch modified kernel with packed queries instead of LUT
  size_t num_pairs = num_queries * nprobe;
  dim3 gridDim(num_pairs, 1, 1);
  dim3 blockDim(256, 1, 1);

  // Recalculate shared memory for new approach
  size_t query_storage = D * sizeof(float);  // For shared query vector
  const int smem_bytes =
    raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(blockDim.x / 32,
                                                                                   MAX_TOP_K);

  // Now we need: packed query bits, candidate storage, and query vector
  // this part is also used to store ip2 results
  size_t packed_query_size =
    max(num_bits * num_words * sizeof(uint32_t), cur_ivf.max_cluster_length * sizeof(float));
  size_t candidate_storage = cur_ivf.max_cluster_length * (sizeof(float) + sizeof(int));
  size_t shared_mem_size   = max(packed_query_size + candidate_storage + query_storage +
                                 10 * sizeof(float),  // +sizeof(float) for width
                               (size_t)smem_bytes);
//    printf("  smem_bytes          : %d bytes (%.2f KB)\n",
//           shared_mem_size, shared_mem_size / 1024.0f);
#ifdef DEBUG_BATCH_SEARCH
  //    printf("num_chunks: %d, candidate_storage: %d, query_storage: %d, smem_bytes: %d\n",
  //    num_chunks, candidate_storage, query_storage, smem_bytes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error Before Launching Distance Kernel! ") +
                             cudaGetErrorString(err));
  }
  printf("shared_mem_size in KB: %lu\n", shared_mem_size / 1024);
#endif

  if (shared_mem_size > 49152) {
    cudaFuncSetAttribute(
      computeInnerProductsWithBitwiseOpt, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  }

  if (!use_4bit) {
    computeInnerProductsWithBitwiseOpt<<<gridDim, blockDim, shared_mem_size, stream>>>(
      d_sorted_pairs,
      d_query,
      cur_ivf.d_short_data,  // This is already transposed bit-packed data
      d_cluster_meta,
      d_packed_queries,  // Packed query bit planes
      d_widths,          // Query scaling factors
      cur_ivf.d_short_factors_batch,
      d_G_k1xSumq,
      d_G_kbxSumq,
      d_centroid_distances,
      topk,
      num_queries,
      nprobe,
      num_pairs,
      cur_ivf.num_centroids,
      D,
      d_topk_threshold_batch,
      15,
      cur_ivf.max_cluster_length,
      cur_ivf.ex_bits,
      cur_ivf.d_long_code,
      reinterpret_cast<const float*>(cur_ivf.d_ex_factor),
      cur_ivf.d_ids,
      d_topk_dists,
      d_topk_pids,
      d_query_write_counters,
      num_bits,  // Add num_bits parameter
      num_words  // Add num_words parameter
    );
  } else {
    //        std::cout << "using 4bit queries" << std::endl;
    computeInnerProductsWithBitwiseOpt4bit<<<gridDim, blockDim, shared_mem_size, stream>>>(
      d_sorted_pairs,
      d_query,
      cur_ivf.d_short_data,  // This is already transposed bit-packed data
      d_cluster_meta,
      d_packed_queries,  // Packed query bit planes
      d_widths,          // Query scaling factors
      cur_ivf.d_short_factors_batch,
      d_G_k1xSumq,
      d_G_kbxSumq,
      d_centroid_distances,
      topk,
      num_queries,
      nprobe,
      num_pairs,
      cur_ivf.num_centroids,
      D,
      d_topk_threshold_batch,
      15,
      cur_ivf.max_cluster_length,
      cur_ivf.ex_bits,
      cur_ivf.d_long_code,
      reinterpret_cast<const float*>(cur_ivf.d_ex_factor),
      cur_ivf.d_ids,
      d_topk_dists,
      d_topk_pids,
      d_query_write_counters,
      num_bits,  // Add num_bits parameter
      num_words  // Add num_words parameter
    );
  }

#ifdef DEBUG_BATCH_SEARCH
  //    printf("num_chunks: %d, candidate_storage: %d, query_storage: %d, smem_bytes: %d\n",
  //    num_chunks, candidate_storage, query_storage, smem_bytes);
  cudaError_t err2 = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error in running distance computation! ") +
                             cudaGetErrorString(err2));
  }
#endif

  // Merge results
  mergeClusterTopKFinal(d_topk_dists,
                        d_topk_pids,
                        d_final_dists,
                        d_final_pids,
                        num_queries,
                        nprobe,
                        topk,
                        true,
                        stream);

  // Cleanup
  cudaFreeAsync(d_topk_threshold_batch, stream);
  cudaFreeAsync(d_query_ranges, stream);
  cudaFreeAsync(d_widths, stream);
  cudaFreeAsync(d_quantized_queries, stream);
  cudaFreeAsync(d_packed_queries, stream);
  cudaFreeAsync(d_query_write_counters, stream);
}

// For two-round kernel search: first round for each query's nearest clusters, in this process we
// compute LUTs and thresholds, then for teh rest cluster-query pairs, in this process we do not
// check LUTs or update thresholds
void SearcherGPU::SearchClusterQueryPairsPreComputeThreshold(
  const IVFGPU& cur_ivf,
  IVFGPU::GPUClusterMeta* d_cluster_meta,
  ClusterQueryPair* d_nearest_sorted_pairs,
  ClusterQueryPair* d_rest_sorted_pairs,
  size_t num_queries,
  const float* d_query,
  const float* d_G_k1xSumq,
  const float* d_G_kbxSumq,
  size_t nprobe,
  size_t topk,
  float* d_topk_dists,  // sizeof(float)*topk*nprobe*query
  PID* d_topk_pids,
  float* d_final_dists,
  PID* d_final_pids,
  cudaStream_t stream)
{
#ifdef DEBUG_BATCH_SEARCH
  // check whether pairs are wrong
//    checkAndPrintNegativeValues(d_sorted_pairs, num_queries, nprobe);
#endif
  // First allocate space for LUT
  size_t lut_size =
    num_queries * (cur_ivf.num_padded_dim / BITS_PER_CHUNK) * LUT_SIZE * sizeof(float);
  // each line's space is (cur_ivf.num_padded_dim / BITS_PER_CHUNK) * LUT_SIZE * sizeof(float);
  float* d_lut_for_queries = nullptr;
  cudaMallocAsync(&d_lut_for_queries, lut_size, stream);
  thrust::fill(thrust::cuda::par.on(stream),
               d_lut_for_queries,
               d_lut_for_queries + (lut_size / sizeof(float)),
               -std::numeric_limits<float>::infinity());  // initially set to INVALID value
  // Clean the input distances
  size_t candidates_per_query = nprobe * topk;
  size_t total_elements       = num_queries * candidates_per_query;
  int threads                 = 256;
  int blocks                  = (total_elements + threads - 1) / threads;

  initDistancesKernel<<<blocks, threads, 0, stream>>>(d_topk_dists, total_elements);
  int* d_query_write_counters;  // One counter per query, indicates where to store final results
                                // (0~nprobe)
  cudaMallocAsync(&d_query_write_counters, num_queries * sizeof(int), stream);
  cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream);  // Initialize to 0

  float* d_topk_threshold_batch;
  cudaMallocAsync(&d_topk_threshold_batch, sizeof(float) * num_queries, stream);
  thrust::fill(thrust::cuda::par.on(stream),
               d_topk_threshold_batch,
               d_topk_threshold_batch + num_queries,
               std::numeric_limits<float>::infinity());
  // Then launch kernel for computation
  size_t num_nearest_pairs = num_queries;
  size_t num_rest_pairs    = num_queries * (nprobe - 1);
  dim3 gridDim(num_nearest_pairs, 1, 1);
  dim3 blockDim(256, 1, 1);
  size_t num_chunks        = D / BITS_PER_CHUNK;
  size_t candidate_storage = cur_ivf.max_cluster_length * (2 * sizeof(float) + sizeof(int));
  size_t query_storage     = D * sizeof(float);  // For shared query vector

  const int smem_bytes =
    raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(blockDim.x / 32,
                                                                                   MAX_TOP_K);
  size_t shared_mem_size =
    num_chunks * LUT_SIZE * sizeof(float) + candidate_storage + query_storage + smem_bytes;
#ifdef DEBUG_BATCH_SEARCH
  printf("num_chunks: %d, candidate_storage: %d, query_storage: %d, smem_bytes: %d\n",
         num_chunks,
         candidate_storage,
         query_storage,
         smem_bytes);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error Before Launching Distance Kernel! ") +
                             cudaGetErrorString(err));
  }
  printf("shared_mem_size in KB: %lu\n", shared_mem_size / 1024);
#endif
  // Note that for large dimensions, we need to set it for specific kernel
  if (shared_mem_size > 49152) {
    // for larger dimensions
#ifdef DEBUG_BATCH_SEARCH
    printf("Using larger shared memory of %d:\n", shared_mem_size);
#endif
    cudaFuncSetAttribute(computeInnerProductsWithLUT,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         98304);  // 96KB for ampere devices
  }
#ifdef DEBUG_BATCH_SEARCH
  printf("ivf.max_cluster_length: %d\n", cur_ivf.max_cluster_length);
#endif
  // first round: compute LUT and search the nearest clusters for each query to get a proper
  // threshold
  computeInnerProductsWithAlwaysLUT<<<gridDim, blockDim, shared_mem_size, stream>>>(
    d_nearest_sorted_pairs,
    d_query,
    cur_ivf.d_short_data,
    d_cluster_meta,
    d_lut_for_queries,
    cur_ivf.d_short_factors_batch,
    d_G_k1xSumq,
    d_G_kbxSumq,
    d_centroid_distances,
    topk,
    num_queries,
    nprobe,
    num_nearest_pairs,
    cur_ivf.num_centroids,
    D,
    d_topk_threshold_batch,
    15,  // by default just set amplification vector to 10
    cur_ivf.max_cluster_length,
    cur_ivf.ex_bits,
    cur_ivf.d_long_code,
    reinterpret_cast<const float*>(cur_ivf.d_ex_factor),
    cur_ivf.d_ids,
    d_topk_dists,
    d_topk_pids,
    d_query_write_counters);
  if (num_rest_pairs > 0) {
    gridDim.x = num_rest_pairs;
    // second round: use previously computed LUT and threshold to search rest pairs
    computeInnerProductsWithLUTWithoutUpdatingThreshold<<<gridDim,
                                                          blockDim,
                                                          shared_mem_size,
                                                          stream>>>(
      d_rest_sorted_pairs,
      d_query,
      cur_ivf.d_short_data,
      d_cluster_meta,
      d_lut_for_queries,
      cur_ivf.d_short_factors_batch,
      d_G_k1xSumq,
      d_G_kbxSumq,
      d_centroid_distances,
      topk,
      num_queries,
      nprobe,
      num_rest_pairs,
      cur_ivf.num_centroids,
      D,
      d_topk_threshold_batch,
      15,  // by default just set amplification vector to 10
      cur_ivf.max_cluster_length,
      cur_ivf.ex_bits,
      cur_ivf.d_long_code,
      reinterpret_cast<const float*>(cur_ivf.d_ex_factor),
      cur_ivf.d_ids,
      d_topk_dists,
      d_topk_pids,
      d_query_write_counters);
  }

#ifdef DEBUG_BATCH_SEARCH
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Error Before Merge! ") + cudaGetErrorString(err));
  }
#endif

  // merge results from different blocks
  mergeClusterTopKFinal(d_topk_dists,
                        d_topk_pids,
                        d_final_dists,
                        d_final_pids,
                        num_queries,
                        nprobe,
                        topk,
                        true,  // sorted=true for ordered results
                        stream);

  //    std::cout << "block distances merged!" << std::endl;
  cudaFreeAsync(d_topk_threshold_batch, stream);
  cudaFreeAsync(d_lut_for_queries, stream);
  cudaFreeAsync(d_query_write_counters, stream);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
