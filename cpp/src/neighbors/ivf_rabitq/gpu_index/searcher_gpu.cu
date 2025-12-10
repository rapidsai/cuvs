/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

#include "../utils/memory.hpp"
#include "searcher_gpu.cuh"

#include <raft/matrix/select_k.cuh>
#include <raft/neighbors/detail/ivf_flat_interleaved_scan.cuh>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <string>
#include <vector>

namespace cuvs::neighbors::ivf_rabitq::detail {

#define MAX_TOP_K               64  // power of 2, as local_topk_capacity, assumes that topk is less than 100
#define MAX_CANDIDATES_PER_PAIR 1000  // suppose topk = 100, M = 10

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16
static constexpr int WARP_SIZE      = 32;

// --- Tunables ---
using T    = float;
using IdxT = uint32_t;

using lut_dtype = __half;  // FP16 alternative

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
  const float* d_threshold,        // NEW: threshold for each query
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
    q_g_add   = d_centroid_distances[query_idx * num_centroids + cluster_idx];
    q_g_error = sqrtf(q_g_add);

    // Get query factor
    q_k1xsumq      = d_G_k1xSumq[query_idx];
    threshold      = d_threshold[query_idx];  // NEW: load threshold
    num_candidates = 0;                       // NEW: initialize counter
  }
  __syncthreads();

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
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_dists[candidate_slot]   = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

  // Step 2 Part 2: Determine which candidates to use
  int final_num_candidates = min(num_candidates, (int)max_candidates_per_pair);

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
        float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
        float f_ex_add     = ex_factors.x;
        float f_ex_rescale = ex_factors.y;

        // Compute final distance using pre-computed ip2
        ex_dist = f_ex_add + q_g_add +
                  f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
        //                ex_dist = ex_dist+1;
        // Get PID
        pid = (uint32_t)d_pids[global_vec_idx];

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

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
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

      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }
  }
}

__global__ void computeInnerProductsWithLUTNoEX(
        const ClusterQueryPair* d_sorted_pairs,
        const float* d_query,
        const uint32_t* d_short_data,
        const IVFGPU::GPUClusterMeta* d_cluster_meta,
        float* d_lut_for_queries,
        const float* d_short_factors,        // NEW
        const float* d_G_k1xSumq,            // NEW
        const float* d_G_kbxSumq,            // NEW (not used yet)
        const float* d_centroid_distances,   // NEW
        uint32_t topk,
        uint32_t num_queries,
        uint32_t nprobe,
        uint32_t num_pairs,
        uint32_t num_centroids,
        uint32_t D,
        const float* d_threshold,            // NEW: threshold for each query
        uint32_t M,                            // NEW: multiplier for topk
        uint32_t max_candidates_per_pair,      // NEW: max storage per pair, 1000 suggested
        uint32_t ex_bits,                     // NEW: bits per dimension in ex codes
        const uint8_t* d_long_code,         // NEW: long codes for all vectors
        const float* d_ex_factor,           // NEW: ex factors for distance computation
        const PID* d_pids,                  // NEW: PIDs for all vectors
        float* d_topk_dists,                // NEW: output top-k distances
        PID* d_topk_pids,                    // NEW: output top-k PIDs
        int* d_query_write_counters
) {
    // Each block handles one <cluster, query> pair
//    const int block_id = blockIdx.x + blockIdx.y * gridDim.x +
//                         blockIdx.z * gridDim.x * gridDim.y;
    const int block_id = blockIdx.x;    // simply use 1-D block

    if (block_id >= num_pairs) return;

    // Get the cluster-query pair for this block
    ClusterQueryPair pair = d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx = pair.query_idx;

    // Check bounds
    if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

    // Get cluster metadata
    size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
    size_t cluster_start_index = d_cluster_meta[cluster_idx].start_index;

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0  && threadIdx.x == 0 ) {
        printf("Preparation completed!\n");
    }
#endif

    // Calculate LUT parameters
    const uint32_t num_chunks = D / BITS_PER_CHUNK;
    const uint32_t lut_per_query_size = num_chunks * LUT_SIZE;

    // Shared memory for LUT
    extern __shared__ __align__(256) char shared_mem_raw[];
    float* shared_lut = reinterpret_cast<float*>(shared_mem_raw);
    // Calculate offset for other shared arrays after BF16 LUT
    uint32_t lut_bytes = num_chunks * LUT_SIZE * sizeof(float);


    // Thread index within the block
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Pointer to this query's LUT in global memory
    float* query_lut = d_lut_for_queries + query_idx * lut_per_query_size;

    // Then Load LUT into shared memory
    // Direct copy of BF16 values
    for (uint32_t i = tid; i < lut_per_query_size; i += num_threads) {
        shared_lut[i] = query_lut[i];
    }

    __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0  && threadIdx.x == 0 ) {
        printf("LUT computation & load finished!\n");
    }
#endif

    // Step 2 Part 1: Compute distances using LUT && decide candidates

    // Shared values for this <cluster, query> pair

    __shared__ int num_candidates;  // counter for candidates
    __shared__ float q_g_add;      // squared distance to centroid
    __shared__ float q_k1xsumq;    // query factor
    __shared__ float q_g_error;    // sqrt(q_g_add)
    __shared__ float threshold;     // threshold for this query
    // Load shared query-cluster values
    if (tid == 0) {
        // Get squared distance from query to this cluster's centroid
        q_g_add = d_centroid_distances[query_idx * num_centroids + cluster_idx];
        q_g_error = sqrtf(q_g_add);

        // Get query factor
        q_k1xsumq = d_G_k1xSumq[query_idx];
        threshold = d_threshold[query_idx];  // NEW: load threshold
        num_candidates = 0;                  // NEW: initialize counter
    }
    __syncthreads();

    // Allocate shared memory for candidate storage (after LUT)
    // Assuming extern shared memory is large enough
    float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
    int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);

    // Calculate short code parameters
    const uint32_t short_code_length = D / 32;  // number of uint32_t per vector
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
            float3 factors = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
            float f_add = factors.x;
            float f_rescale = factors.y;
            float f_error = factors.z;

            // Compute inner product using LUT
            float ip = 0.0f;

            // Process each uint32_t of the short code
            for (uint32_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
                // Access short code in transposed layout
                // For transposed layout: vec1[dim0-31], vec2[dim0-31], ..., vecn[dim0-31]
                size_t short_code_offset = cluster_start_index * short_code_length +
                                           uint32_idx * num_vectors_in_cluster +
                                           vec_idx;
                uint32_t short_code_chunk = d_short_data[short_code_offset];

                // Process 8 4-bit chunks from this uint32_t
                for (int chunk_in_uint32 = 0; chunk_in_uint32 < chunks_per_uint32; chunk_in_uint32++) {
                    // Extract 4-bit pattern
                    // Note: in uint32_t, lowest dim is at bit 31 (MSB)
                    // So we extract from high bits to low bits
                    int shift = 28 - (chunk_in_uint32 * BITS_PER_CHUNK);  // 28, 24, 20, 16, 12, 8, 4, 0
                    int pattern = (short_code_chunk >> shift) & 0xF;  // Extract 4 bits

                    // Look up in LUT
                    uint32_t lut_chunk_idx = uint32_idx * chunks_per_uint32 + chunk_in_uint32;
                    uint32_t lut_offset = lut_chunk_idx * LUT_SIZE + pattern;

                    // Accumulate inner product
//                    ip += __bfloat162float(shared_lut_bf16[lut_offset]);
                    ip += shared_lut[lut_offset];
                }
            }

            // Compute estimated distance
            final_1bit_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

            // Check threshold
            if (final_1bit_dist < threshold) {
                is_candidate = true;
                final_1bit_pid = d_pids[cluster_start_index + vec_idx];
            }
        }
        // Collectively add candidates to shared memory
        __syncwarp();  // Sync within warp for atomics

        if (is_candidate) {
            int candidate_slot = atomicAdd(&num_candidates, 1);

            if (candidate_slot < max_candidates_per_pair) {
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
            using block_sort_t = typename raft::neighbors::ivf_flat::detail::flat_block_sort<
                    MAX_TOP_K, true, float, uint32_t>::type;
            block_sort_t queue(topk);

            // Additional shared values needed for Step 3
            __shared__ float q_kbxsumq;
            if (tid == 0) {
                q_kbxsumq = d_G_kbxSumq[query_idx];
            }
            __syncthreads();

            const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

            // Each warp processes different candidates
            for (int c = 0; c < candidates_per_thread; ++c) {
                int cand_idx = tid + c * num_threads;

                if (cand_idx < num_candidates && cand_idx < max_candidates_per_pair) {
                    final_1bit_dist = shared_candidate_ips[cand_idx];
                    final_1bit_pid = shared_candidate_indices[cand_idx];
                }
                else {
                    final_1bit_dist = INFINITY;
                    final_1bit_pid = 0;
                }
                queue.add(final_1bit_dist, final_1bit_pid);
            }

            __syncthreads();


            // Step 3: Merge results and write back top-k

            queue.done((uint8_t*) shared_lut);

            // Atomically get write position
            if (tid == 0) {
                probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1);
            }
            __syncthreads();

            if (probe_slot >= nprobe) {
//            printf("Impossible!!!!!!!\n");
                return;
            }

            // Calculate output offset and store results
            uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
            queue.store(d_topk_dists + output_offset,
                        (uint32_t*) (d_topk_pids + output_offset));
        }


        // Step 4: Update threshold atomically (simplified version)
        // If threshold only decreases (gets tighter), we can use atomicMin

        float max_topk_dist;

        if (tid == 0) {
            max_topk_dist = -INFINITY;

            // Find the maximum distance in our top-k results
            uint32_t output_offset = query_idx * (topk * nprobe) +
                                   probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

            for (uint32_t i = 0; i < topk; i++) {
                float dist = d_topk_dists[output_offset + i];
                if (dist > 0 && dist > max_topk_dist) {
                    max_topk_dist = dist;
                }
            }
        }

        __syncthreads();

        // Update threshold using atomicMin (for floats)
        // max_topk_dist should be > 0 to prevent using initialized memory
        if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
            // Use integer interpretation for atomic operations
            int* threshold_ptr = (int*) (d_threshold + query_idx);
            int new_val = __float_as_int(max_topk_dist);

            // Atomic minimum for floats (assuming positive distances)
            atomicMin(threshold_ptr, new_val);

            // Note: atomicMin on int representation works correctly for positive floats
            // because IEEE 754 float format preserves ordering for positive values
        }
    }
}

// optimize loops and data types
__global__ void computeInnerProductsWithLUT16Opt(
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
          ip += __half2float(shared_lut_bf16[lut_offset]);
        }
      }

      // Compute estimated distance
      float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

      // Compute lower bound
      float low_dist = est_dist - f_error * q_g_error;

      // Check threshold
      if (low_dist < threshold) {
        is_candidate = true;
        //                local_low_dist = est_dist;
        local_ip = ip;
      }
    }
    // Collectively add candidates to shared memory
    __syncwarp();  // Sync within warp for atomics

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < max_candidates_per_pair) {
        //                shared_candidate_dists[candidate_slot] = local_low_dist;
        shared_candidate_ips[candidate_slot]     = local_ip;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  __syncthreads();

  // Step 2 Part 2: Determine which candidates to use
  //    int final_num_candidates = min(num_candidates, (int)max_candidates_per_pair);

  // Step 3 opt: Compute more accurate distances and select top-k
  // Opt: warp-level dist and then thread-level ex dist restore

  __syncthreads();
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
          float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
          // Get PID
          pid = (uint32_t)d_pids[global_vec_idx];

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

      queue.done((uint8_t*)shared_lut_bf16);

      // Atomically get write position
      if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= nprobe) { return; }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));
    }

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
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

      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }
  }
}

__global__ void computeInnerProductsWithLUT16OptNoEX(
        const ClusterQueryPair* d_sorted_pairs,
        const float* d_query,
        const uint32_t* d_short_data,
        const IVFGPU::GPUClusterMeta* d_cluster_meta,
        lut_dtype* d_lut_for_queries,
        const float* d_short_factors,        // NEW
        const float* d_G_k1xSumq,            // NEW
        const float* d_G_kbxSumq,            // NEW (not used yet)
        const float* d_centroid_distances,   // NEW
        uint32_t topk,
        uint32_t num_queries,
        uint32_t nprobe,
        uint32_t num_pairs,
        uint32_t num_centroids,
        uint32_t D,
        const float* d_threshold,            // NEW: threshold for each query
        uint32_t M,                            // NEW: multiplier for topk
        uint32_t max_candidates_per_pair,      // NEW: max storage per pair, 1000 suggested
        uint32_t ex_bits,                     // NEW: bits per dimension in ex codes
        const uint8_t* d_long_code,         // NEW: long codes for all vectors
        const float* d_ex_factor,           // NEW: ex factors for distance computation
        const PID* d_pids,                  // NEW: PIDs for all vectors
        float* d_topk_dists,                // NEW: output top-k distances
        PID* d_topk_pids,                    // NEW: output top-k PIDs
        int* d_query_write_counters
) {
    // Each block handles one <cluster, query> pair
//    const int block_id = blockIdx.x + blockIdx.y * gridDim.x +
//                         blockIdx.z * gridDim.x * gridDim.y;
    const int block_id = blockIdx.x;    // simply use 1-D block

    if (block_id >= num_pairs) return;

    // Get the cluster-query pair for this block
    ClusterQueryPair pair = d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx = pair.query_idx;

    // Check bounds
    if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

    // Get cluster metadata
    size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
    size_t cluster_start_index = d_cluster_meta[cluster_idx].start_index;

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0  && threadIdx.x == 0 ) {
        printf("Preparation completed!\n");
    }
#endif

    // Calculate LUT parameters
    const uint32_t num_chunks = D / BITS_PER_CHUNK;
    const uint32_t lut_per_query_size = num_chunks * LUT_SIZE;

    // Shared memory for LUT
    extern __shared__ __align__(256) char shared_mem_raw[];
    lut_dtype* shared_lut_bf16 = reinterpret_cast<lut_dtype*>(shared_mem_raw);
    // Calculate offset for other shared arrays after BF16 LUT
    uint32_t lut_bytes = num_chunks * LUT_SIZE * sizeof(lut_dtype);


    // Thread index within the block
    const int tid = threadIdx.x;
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
    if (blockIdx.x == 0  && threadIdx.x == 0 ) {
        printf("LUT computation & load finished!\n");
    }
#endif

    // Step 2 Part 1: Compute distances using LUT && decide candidates

    // Shared values for this <cluster, query> pair

    __shared__ int num_candidates;  // counter for candidates
    __shared__ float q_g_add;      // squared distance to centroid
    __shared__ float q_k1xsumq;    // query factor
    __shared__ float q_g_error;    // sqrt(q_g_add)
    __shared__ float threshold;     // threshold for this query
    // Load shared query-cluster values
    if (tid == 0) {
        // Get squared distance from query to this cluster's centroid
        q_g_add = d_centroid_distances[query_idx * num_centroids + cluster_idx];
        q_g_error = sqrtf(q_g_add);

        // Get query factor
        q_k1xsumq = d_G_k1xSumq[query_idx];
        threshold = d_threshold[query_idx];  // NEW: load threshold
        num_candidates = 0;                  // NEW: initialize counter
    }
    __syncthreads();

    // Allocate shared memory for candidate storage (after LUT)
    // Assuming extern shared memory is large enough
    float* shared_candidate_ips;
    shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
    int* shared_candidate_indices = (int*)(shared_candidate_ips + max_candidates_per_pair);

    // Calculate short code parameters
    const uint32_t short_code_length = D / 32;  // number of uint32_t per vector
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
            float3 factors = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
            float f_add = factors.x;
            float f_rescale = factors.y;
            float f_error = factors.z;

            // Compute inner product using LUT
            float ip = 0.0f;

            // Process each uint32_t of the short code
            for (uint32_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
                // Access short code in transposed layout
                // For transposed layout: vec1[dim0-31], vec2[dim0-31], ..., vecn[dim0-31]
                size_t short_code_offset = cluster_start_index * short_code_length +
                                           uint32_idx * num_vectors_in_cluster +
                                           vec_idx;
                uint32_t short_code_chunk = d_short_data[short_code_offset];

                // Process 8 4-bit chunks from this uint32_t
                for (int chunk_in_uint32 = 0; chunk_in_uint32 < chunks_per_uint32; chunk_in_uint32++) {
                    // Extract 4-bit pattern
                    // Note: in uint32_t, lowest dim is at bit 31 (MSB)
                    // So we extract from high bits to low bits
                    int shift = 28 - (chunk_in_uint32 * BITS_PER_CHUNK);  // 28, 24, 20, 16, 12, 8, 4, 0
                    int pattern = (short_code_chunk >> shift) & 0xF;  // Extract 4 bits

                    // Look up in LUT
                    uint32_t lut_chunk_idx = uint32_idx * chunks_per_uint32 + chunk_in_uint32;
                    uint32_t lut_offset = lut_chunk_idx * LUT_SIZE + pattern;

                    // Accumulate inner product
//                    ip += __bfloat162float(shared_lut_bf16[lut_offset]);
                    ip += __half2float(shared_lut_bf16[lut_offset]);
                }
            }

            // Compute estimated distance
            final_1bit_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

            // Check threshold
            if (final_1bit_dist < threshold) {
                is_candidate = true;
                final_1bit_pid = d_pids[cluster_start_index + vec_idx];
            }
        }
        // Collectively add candidates to shared memory
        __syncwarp();  // Sync within warp for atomics

        if (is_candidate) {
            int candidate_slot = atomicAdd(&num_candidates, 1);

            if (candidate_slot < max_candidates_per_pair) {
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
            using block_sort_t = typename raft::neighbors::ivf_flat::detail::flat_block_sort<
                    MAX_TOP_K, true, float, uint32_t>::type;
            block_sort_t queue(topk);

            // Additional shared values needed for Step 3
            __shared__ float q_kbxsumq;
            if (tid == 0) {
                q_kbxsumq = d_G_kbxSumq[query_idx];
            }
            __syncthreads();

            const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

            // Each warp processes different candidates
            for (int c = 0; c < candidates_per_thread; ++c) {
                int cand_idx = tid + c * num_threads;

                if (cand_idx < num_candidates && cand_idx < max_candidates_per_pair) {
                    final_1bit_dist = shared_candidate_ips[cand_idx];
                    final_1bit_pid = shared_candidate_indices[cand_idx];
                }
                else {
                    final_1bit_dist = INFINITY;
                    final_1bit_pid = 0;
                }
                queue.add(final_1bit_dist, final_1bit_pid);
            }

            __syncthreads();


            // Step 3: Merge results and write back top-k

            queue.done((uint8_t*) shared_lut_bf16);

            // Atomically get write position
            if (tid == 0) {
                probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1);
            }
            __syncthreads();

            if (probe_slot >= nprobe) {
//            printf("Impossible!!!!!!!\n");
                return;
            }

            // Calculate output offset and store results
            uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
            queue.store(d_topk_dists + output_offset,
                        (uint32_t*) (d_topk_pids + output_offset));
        }


        // Step 4: Update threshold atomically (simplified version)
        // If threshold only decreases (gets tighter), we can use atomicMin

        float max_topk_dist;

        if (tid == 0) {
            max_topk_dist = -INFINITY;

            // Find the maximum distance in our top-k results
            uint32_t output_offset = query_idx * (topk * nprobe) +
                                   probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

            for (uint32_t i = 0; i < topk; i++) {
                float dist = d_topk_dists[output_offset + i];
                if (dist > 0 && dist > max_topk_dist) {
                    max_topk_dist = dist;
                }
            }
        }

        __syncthreads();

        // Update threshold using atomicMin (for floats)
        // max_topk_dist should be > 0 to prevent using initialized memory
        if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
            // Use integer interpretation for atomic operations
            int* threshold_ptr = (int*) (d_threshold + query_idx);
            int new_val = __float_as_int(max_topk_dist);

            // Atomic minimum for floats (assuming positive distances)
            atomicMin(threshold_ptr, new_val);

            // Note: atomicMin on int representation works correctly for positive floats
            // because IEEE 754 float format preserves ordering for positive values
        }
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
      }
    }

    __syncwarp();

    if (is_candidate) {
      int candidate_slot = atomicAdd(&num_candidates, 1);
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = local_ip_quantized;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }

  __syncthreads();

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
          float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);
          //
          // Get PID
          pid = (uint32_t)d_pids[global_vec_idx];

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
      if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= nprobe) { return; }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));
    }

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
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

      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt8bitNoEX(
        const ClusterQueryPair* d_sorted_pairs,
        const float* d_query,
        const uint32_t* d_short_data,           // Transposed bit-packed data
        const IVFGPU::GPUClusterMeta* d_cluster_meta,
        const uint32_t* d_packed_queries,       // Packed query bit planes
        const float* d_widths,                  // Query scaling factors
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
        uint32_t num_bits,                      // Added: number of bits (8 for int8)
        uint32_t num_words                      // Added: D/32
) {
    const int block_id = blockIdx.x;
    if (block_id >= num_pairs) return;

    ClusterQueryPair pair = d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx = pair.query_idx;

    if (cluster_idx >= num_centroids || query_idx >= num_queries) return;

    size_t num_vectors_in_cluster = d_cluster_meta[cluster_idx].num;
    size_t cluster_start_index = d_cluster_meta[cluster_idx].start_index;

    // Shared memory layout
    extern __shared__ __align__(256) char shared_mem_raw_2[];

    // Load packed query bit planes into shared memory
    uint32_t* shared_packed_query = reinterpret_cast<uint32_t*>(shared_mem_raw_2);

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Load this query's packed bit planes
    const uint32_t* query_packed_ptr = d_packed_queries + query_idx * num_bits * num_words;
    for (uint32_t i = tid; i < num_bits * num_words; i += num_threads) {
        shared_packed_query[i] = query_packed_ptr[i];
    }

    // Load query width
    __shared__ float query_width;
    if (tid == 0) {
        query_width = d_widths[query_idx];
    }
//    float query_width = d_widths[query_idx];
    __syncthreads();

    // Shared values for this <cluster, query> pair
    __shared__ int num_candidates;
    __shared__ float q_g_add;
    __shared__ float q_k1xsumq;
    __shared__ float q_g_error;
    __shared__ float threshold;

    if (tid == 0) {
        q_g_add = d_centroid_distances[query_idx * num_centroids + cluster_idx];
        q_g_error = sqrtf(q_g_add);
        q_k1xsumq = d_G_k1xSumq[query_idx];
        threshold = d_threshold[query_idx];
        num_candidates = 0;
    }
    __syncthreads();

    // Allocate shared memory for candidates
    size_t packed_query_bytes = max (num_bits * num_words * sizeof(uint32_t), max_candidates_per_pair * sizeof(float));
    float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw_2 + packed_query_bytes);
    int* shared_candidate_indices = reinterpret_cast<int*>(shared_candidate_ips + max_candidates_per_pair);
    float* shared_query = (float*) (shared_candidate_indices + max_candidates_per_pair);
    const size_t short_code_length = D / 32;
    // Step 2 Part 1: Compute bitwise inner products
    const int vectors_per_iteration = num_threads;

    // Ori verison --------------------------------------
    // Optimized first-round IP computation - accumulate on the fly
    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += vectors_per_iteration) {


        size_t vec_idx = vec_base + tid;

        bool is_candidate = false;
        float local_ip_quantized = 0;

        if (vec_idx < num_vectors_in_cluster) {
            size_t factor_offset = cluster_start_index + vec_idx;
            float3 factors = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
            float f_add = factors.x;
            float f_rescale = factors.y;
            float f_error = factors.z;

            int32_t accumulator = 0;  // Single accumulator, no array needed

            // Load data once, accumulate directly
//            int32_t accumulator2 = 0;
//#pragma unroll 4
            for (int word = 0; word < num_words; ++word) {
                size_t data_offset = cluster_start_index * num_words +
                                     word * num_vectors_in_cluster + vec_idx;
                uint32_t data_word = d_short_data[data_offset];

                accumulator += __popc(shared_packed_query[0 * num_words + word] & data_word) << 0;
                accumulator += __popc(shared_packed_query[1 * num_words + word] & data_word) << 1;
                accumulator += __popc(shared_packed_query[2 * num_words + word] & data_word) << 2;
                accumulator += __popc(shared_packed_query[3 * num_words + word] & data_word) << 3;
                accumulator += __popc(shared_packed_query[4 * num_words + word] & data_word) << 4;
                accumulator += __popc(shared_packed_query[5 * num_words + word] & data_word) << 5;
                accumulator += __popc(shared_packed_query[6 * num_words + word] & data_word) << 6;
                accumulator -= __popc(shared_packed_query[7 * num_words + word] & data_word) << 7;  // Sign bit
            }

            // Restore scale and compute estimated distance
            float ip = (float) accumulator * query_width;
            float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
            float low_dist = est_dist - f_error * q_g_error;


            if (low_dist < threshold) {
                is_candidate = true;
                local_ip_quantized = ip;

#ifdef DEBUG_BATCH_SEARCH
                //                local_ip_quantized = est_dist; //debug
//                printf("low distance : %f, local_ip_quantized: %f \n", low_dist, local_ip_quantized);
#endif
            }
        }

        __syncwarp();

        if (is_candidate) {
            int candidate_slot = atomicAdd(&num_candidates, 1);
            if (candidate_slot < max_candidates_per_pair) {
                shared_candidate_ips[candidate_slot] = local_ip_quantized;
                shared_candidate_indices[candidate_slot] = vec_idx;
            }
        }
#ifdef DEBUG_BATCH_SEARCH
        //        if (threadIdx.x == 0) {
//            printf("num_vectors in cluster: %d, vec %d finished.\n", num_vectors_in_cluster, vec_idx);
//        }
#endif
    }
    // -----------------

    __syncthreads();

#ifdef DEBUG_BATCH_SEARCH
    if (blockIdx.x == 0  && threadIdx.x == 0 ) {
        printf("1bit estimated distance computation finished!\n");
        printf("final_num_candidates_before: %d\n", num_candidates);
    }
#endif

    if (num_candidates > 0) {

        using block_sort_t = typename raft::neighbors::ivf_flat::detail::flat_block_sort<
                MAX_TOP_K, true, float, uint32_t>::type;
        block_sort_t queue(topk);

        for (size_t i = tid; i < D; i += num_threads) {
            shared_query[i] = d_query[query_idx * D + i];
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

            if (cand_idx < num_candidates && cand_idx < max_candidates_per_pair) {
                int vec_idx = shared_candidate_indices[cand_idx];
                size_t factor_offset = cluster_start_index + vec_idx;
                float3 factors = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
                float f_add = factors.x;
                float f_rescale = factors.y;
                float f_error = factors.z;
                size_t global_vec_idx = cluster_start_index + vec_idx;

                // Compute exact inner product with float query
                float exact_ip = 0.0f;

                // Process each uint32_t of the short code
                for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
                    // Access short code in transposed layout
                    size_t short_code_offset = cluster_start_index * short_code_length +
                                               uint32_idx * num_vectors_in_cluster +
                                               vec_idx;
                    uint32_t short_code_chunk = d_short_data[short_code_offset];

                    // Process each bit in the uint32_t
                    // Note: bit 31 is lowest dimension, bit 0 is highest
#pragma unroll 8
                    for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
                        size_t dim = uint32_idx * 32 + bit_idx;
                        if (dim < D) {
                            // Extract bit from MSB to LSB
                            int bit_position = 31 - bit_idx;
                            bool bit_value = (short_code_chunk >> bit_position) & 0x1;

                            // If bit is 1, add the query value
                            if (bit_value) {
                                exact_ip += shared_query[dim];
                            }
                        }
                    }
                }

                // get final results and push to queue
                final_1bit_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
                final_1bit_pid = (uint32_t) d_pids[global_vec_idx];

            }
            else {
                    final_1bit_dist = INFINITY;
                    final_1bit_pid = 0;
            };
            queue.add(final_1bit_dist, final_1bit_pid);
        }

        __syncthreads();
//    ------------------
        __shared__ int probe_slot;
        queue.done((uint8_t*) shared_mem_raw_2);

        // Atomically get write position
        if (tid == 0) {
            probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1);
        }
        __syncthreads();


        // Calculate output offset and store results
        uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
        queue.store(d_topk_dists + output_offset,
                    (uint32_t*) (d_topk_pids + output_offset));


        float max_topk_dist;

        if (tid == 0) {
            max_topk_dist = -INFINITY;

            // Find the maximum distance in our top-k results
            uint32_t output_offset = query_idx * (topk * nprobe) +
                                     probe_slot * topk;  // <-- Use probe_slot, not (block_id % nprobe)

            for (uint32_t i = 0; i < topk; i++) {
                float dist = d_topk_dists[output_offset + i];
                if (dist > 0 && dist > max_topk_dist) {
                    max_topk_dist = dist;
                }
            }
        }

        __syncthreads();

        // Update threshold using atomicMin (for floats)
        // max_topk_dist should be > 0 to prevent using initialized memory
        if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
            // Use integer interpretation for atomic operations
            int* threshold_ptr = (int*) (d_threshold + query_idx);
            int new_val = __float_as_int(max_topk_dist);

            // Atomic minimum for floats (assuming positive distances)
            atomicMin(threshold_ptr, new_val);


            // Note: atomicMin on int representation works correctly for positive floats
            // because IEEE 754 float format preserves ordering for positive values
        }
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
      for (int word = 0; word < num_words; ++word) {
        size_t data_offset =
          cluster_start_index * num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = d_short_data[data_offset];
        accumulator2 += __popc(data_word);

        accumulator += __popc(shared_packed_query[0 * num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * num_words + word] & data_word) << 2;
        accumulator -= __popc(shared_packed_query[3 * num_words + word] & data_word)
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
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = local_ip_quantized;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  // -----------------

  __syncthreads();

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
          float2 ex_factors  = reinterpret_cast<const float2*>(d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          // Compute final distance using pre-computed ip2
          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << ex_bits) * ip + ip2 + q_kbxsumq);

          // Get PID
          pid = (uint32_t)d_pids[global_vec_idx];

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
      if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= nprobe) { return; }

      // Calculate output offset and store results
      uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
      queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));
    }

    // Step 4: Update threshold atomically (simplified version)
    // If threshold only decreases (gets tighter), we can use atomicMin
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

      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
    }
  }
}

__global__ void computeInnerProductsWithBitwiseOpt4bitNoEX(
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
      for (int word = 0; word < num_words; ++word) {
        size_t data_offset =
          cluster_start_index * num_words + word * num_vectors_in_cluster + vec_idx;
        uint32_t data_word = d_short_data[data_offset];

        accumulator += __popc(shared_packed_query[0 * num_words + word] & data_word) << 0;
        accumulator += __popc(shared_packed_query[1 * num_words + word] & data_word) << 1;
        accumulator += __popc(shared_packed_query[2 * num_words + word] & data_word) << 2;
        accumulator -= __popc(shared_packed_query[3 * num_words + word] & data_word)
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
      if (candidate_slot < max_candidates_per_pair) {
        shared_candidate_ips[candidate_slot]     = local_ip_quantized;
        shared_candidate_indices[candidate_slot] = vec_idx;
      }
    }
  }
  // -----------------

  __syncthreads();

  if (num_candidates > 0) {
    using block_sort_t = typename raft::neighbors::ivf_flat::detail::
      flat_block_sort<MAX_TOP_K, true, float, uint32_t>::type;
    block_sort_t queue(topk);

    for (size_t i = tid; i < D; i += num_threads) {
      shared_query[i] = d_query[query_idx * D + i];
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

      if (cand_idx < num_candidates && cand_idx < max_candidates_per_pair) {
        int vec_idx           = shared_candidate_indices[cand_idx];
        size_t factor_offset  = cluster_start_index + vec_idx;
        float3 factors        = reinterpret_cast<const float3*>(d_short_factors)[factor_offset];
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

        // get final results and push to queue
        final_1bit_dist = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
        final_1bit_pid  = (uint32_t)d_pids[global_vec_idx];

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
    if (tid == 0) { probe_slot = atomicAdd(&d_query_write_counters[query_idx], 1); }
    __syncthreads();

    // Calculate output offset and store results
    uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
    queue.store(d_topk_dists + output_offset, (uint32_t*)(d_topk_pids + output_offset));

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

      // Note: atomicMin on int representation works correctly for positive floats
      // because IEEE 754 float format preserves ordering for positive values
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
  dim3 gridDim(num_pairs, 1, 1);
  dim3 blockDim(256, 1, 1);
  size_t num_chunks = D / BITS_PER_CHUNK;
  size_t candidate_storage =
    cur_ivf.get_max_cluster_length() * (2 * sizeof(float) + sizeof(int));  // ip, idx
  size_t query_storage = D * sizeof(float);  // For shared query vector
  const int smem_bytes =
    raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(blockDim.x / 32,
                                                                                   MAX_TOP_K);
  if (cur_ivf.get_ex_bits() != 0) {
    size_t shared_mem_size = num_chunks * LUT_SIZE * sizeof(float) + candidate_storage + query_storage + smem_bytes;
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
    computeInnerProductsWithLUT, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    computeInnerProductsWithLUT<<<gridDim, blockDim, shared_mem_size, stream_>>>(
      d_sorted_pairs,
      d_query,
      cur_ivf.get_short_data_device(),
      d_cluster_meta,
      d_lut_for_queries,
      cur_ivf.get_short_factors_batch_device(),
      d_G_k1xSumq,
      d_G_kbxSumq,
      get_centroid_distances(),
      topk,
      num_queries,
      nprobe,
      num_pairs,
      cur_ivf.get_num_centroids(),
      D,
      d_topk_threshold_batch,
      15,  // by default just set amplification vector to 10
      cur_ivf.get_max_cluster_length(),
      cur_ivf.get_ex_bits(),
      cur_ivf.get_long_code_device(),
      reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
      cur_ivf.get_ids_device(),
      d_topk_dists,
      d_topk_pids,
      d_query_write_counters);
  }
  else {
    size_t shared_mem_size = max(num_chunks * LUT_SIZE * sizeof(float) + cur_ivf.get_max_cluster_length() * (sizeof(float) + sizeof(int)), (size_t)smem_bytes);
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
    computeInnerProductsWithLUTNoEX, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    computeInnerProductsWithLUTNoEX<<<gridDim, blockDim, shared_mem_size, stream_>>>(
      d_sorted_pairs,
      d_query,
      cur_ivf.get_short_data_device(),
      d_cluster_meta,
      d_lut_for_queries,
      cur_ivf.get_short_factors_batch_device(),
      d_G_k1xSumq,
      d_G_kbxSumq,
      get_centroid_distances(),
      topk,
      num_queries,
      nprobe,
      num_pairs,
      cur_ivf.get_num_centroids(),
      D,
      d_topk_threshold_batch,
      15,  // by default just set amplification vector to 10
      cur_ivf.get_max_cluster_length(),
      cur_ivf.get_ex_bits(),
      cur_ivf.get_long_code_device(),
      reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
      cur_ivf.get_ids_device(),
      d_topk_dists,
      d_topk_pids,
      d_query_write_counters);
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

      size_t lut_offset     = chunk_idx * LUT_SIZE + lut_entry;
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
                               cudaStream_t stream,
                               bool use_optimized = false)
{
  dim3 gridDim(num_queries, 1, 1);
  dim3 blockDim(256, 1, 1);

  if (use_optimized) {
    size_t shared_mem_size = (BITS_PER_CHUNK + LUT_SIZE) * sizeof(float);
    precomputeAllLUTs_bf16_optimized<<<gridDim, blockDim, shared_mem_size, stream>>>(
      d_query, d_lut_for_queries, num_queries, D);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {
    precomputeAllLUTs_bf16_simple<<<gridDim, blockDim, 0, stream>>>(
      d_query, d_lut_for_queries, num_queries, D);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
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
    int code_val = __float2int_rn((const_scaling_factor * val) /*+ 0.5*/);  // 4 drop and 5 in
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
  PID* d_final_pids)
{
  // Using BF16 for storage

  // Allocate space for LUT with reduced precision
  size_t lut_elements = num_queries * (cur_ivf.get_num_padded_dim() / BITS_PER_CHUNK) * LUT_SIZE;
  size_t lut_size     = lut_elements * sizeof(lut_dtype);

  lut_dtype* d_lut_for_queries = nullptr;
  RAFT_CUDA_TRY(cudaMallocAsync(&d_lut_for_queries, lut_size, stream_));

  // Initialize with -infinity (convert to BF16)
  float neg_inf          = -std::numeric_limits<float>::infinity();
  lut_dtype neg_inf_bf16 = __float2half(neg_inf);
  // Fill using thrust with BF16 value
  thrust::fill(thrust::cuda::par.on(stream_),
               d_lut_for_queries,
               d_lut_for_queries + lut_elements,
               neg_inf_bf16);

  // Precompute LUTs
  launchPrecomputeLUTs_bf16(
    d_query, d_lut_for_queries, num_queries, cur_ivf.get_num_padded_dim(), stream_);
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
  dim3 gridDim(num_pairs, 1, 1);
  dim3 blockDim(256, 1, 1);
  const int smem_bytes =
    raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(blockDim.x / 32,
                                                                                   MAX_TOP_K);

  if (cur_ivf.get_ex_bits() != 0) {
    size_t query_storage = D * sizeof(float);  // For shared query vector
    size_t first_part_shared_mem =
    max(lut_size / num_queries, cur_ivf.get_max_cluster_length() * (sizeof(float)));
    size_t second_part_shared_mem = cur_ivf.get_max_cluster_length() * (sizeof(float) + sizeof(int));
    size_t third_part_shared_mem  = query_storage;
    // smem reuses first 3 parts
    size_t shared_mem_size =
      max(first_part_shared_mem + second_part_shared_mem + third_part_shared_mem, (size_t)smem_bytes);
    // Note that for large dimensions, we need to set it for specific kernel
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
    computeInnerProductsWithLUT16Opt, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    computeInnerProductsWithLUT16Opt<<<gridDim, blockDim, shared_mem_size, stream_>>>(
      d_sorted_pairs,
      d_query,
      cur_ivf.get_short_data_device(),
      d_cluster_meta,
      d_lut_for_queries,
      cur_ivf.get_short_factors_batch_device(),
      d_G_k1xSumq,
      d_G_kbxSumq,
      get_centroid_distances(),
      topk,
      num_queries,
      nprobe,
      num_pairs,
      cur_ivf.get_num_centroids(),
      D,
      d_topk_threshold_batch,
      15,  // by default just set amplification vector to 10
      cur_ivf.get_max_cluster_length(),
      cur_ivf.get_ex_bits(),
      cur_ivf.get_long_code_device(),
      reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
      cur_ivf.get_ids_device(),
      d_topk_dists,
      d_topk_pids,
      d_query_write_counters);
  }
  else {
    size_t first_part_shared_mem = lut_size / num_queries;
    size_t second_part_shared_mem = cur_ivf.get_max_cluster_length() * (sizeof(float) + sizeof(int));
    // smem reuses first 3 parts
    size_t shared_mem_size =
      max(first_part_shared_mem + second_part_shared_mem, (size_t)smem_bytes);
    // Note that for large dimensions, we need to set it for specific kernel
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
    computeInnerProductsWithLUT16OptNoEX, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    computeInnerProductsWithLUT16OptNoEX<<<gridDim, blockDim, shared_mem_size, stream_>>>(
      d_sorted_pairs,
      d_query,
      cur_ivf.get_short_data_device(),
      d_cluster_meta,
      d_lut_for_queries,
      cur_ivf.get_short_factors_batch_device(),
      d_G_k1xSumq,
      d_G_kbxSumq,
      get_centroid_distances(),
      topk,
      num_queries,
      nprobe,
      num_pairs,
      cur_ivf.get_num_centroids(),
      D,
      d_topk_threshold_batch,
      15,  // by default just set amplification vector to 10
      cur_ivf.get_max_cluster_length(),
      cur_ivf.get_ex_bits(),
      cur_ivf.get_long_code_device(),
      reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
      cur_ivf.get_ids_device(),
      d_topk_dists,
      d_topk_pids,
      d_query_write_counters);
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

  //    std::cout << "block distances merged!" << std::endl;
  RAFT_CUDA_TRY(cudaFreeAsync(d_topk_threshold_batch, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_lut_for_queries, stream_));
  RAFT_CUDA_TRY(cudaFreeAsync(d_query_write_counters, stream_));

  raft::resource::sync_stream(handle_);
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
  // query quantize
  const int num_bits  = use_4bit ? 4 : 8;  // Choose bit width
  const int num_words = (cur_ivf.get_num_padded_dim() + 31) / 32;

  // Allocate memory for quantization
  size_t ranges_size     = num_queries * 2 * sizeof(float);
  size_t widths_size     = num_queries * sizeof(float);
  size_t quantized_size  = num_queries * cur_ivf.get_num_padded_dim() * sizeof(int8_t);
  size_t packed_size     = num_queries * num_bits * num_words * sizeof(uint32_t);
  size_t counters_size   = num_queries * sizeof(int);
  size_t thresholds_size = num_queries * sizeof(float);

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
  size_t candidates_per_query = nprobe * topk;
  size_t total_elements       = num_queries * candidates_per_query;
  int threads                 = 256;
  int blocks                  = (total_elements + threads - 1) / threads;
  initDistancesKernel<<<blocks, threads, 0, stream_>>>(d_topk_dists, total_elements);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream_));

  thrust::fill(thrust::cuda::par.on(stream_),
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
    max(num_bits * num_words * sizeof(uint32_t), cur_ivf.get_max_cluster_length() * sizeof(float));
  size_t candidate_storage = cur_ivf.get_max_cluster_length() * (sizeof(float) + sizeof(int));
  size_t shared_mem_size   = max(packed_query_size + candidate_storage + query_storage +
                                 10 * sizeof(float),  // +sizeof(float) for width
                               (size_t)smem_bytes);

  if (shared_mem_size > 49152) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(
      computeInnerProductsWithBitwiseOpt, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
  }

  if (!use_4bit) {
    if (cur_ivf.get_ex_bits() != 0) {
      computeInnerProductsWithBitwiseOpt<<<gridDim, blockDim, shared_mem_size, stream_>>>(
        d_sorted_pairs,
        d_query,
        cur_ivf.get_short_data_device(),  // This is already transposed bit-packed data
        d_cluster_meta,
        d_packed_queries,  // Packed query bit planes
        d_widths,          // Query scaling factors
        cur_ivf.get_short_factors_batch_device(),
        d_G_k1xSumq,
        d_G_kbxSumq,
        get_centroid_distances(),
        topk,
        num_queries,
        nprobe,
        num_pairs,
        cur_ivf.get_num_centroids(),
        D,
        d_topk_threshold_batch,
        15,
        cur_ivf.get_max_cluster_length(),
        cur_ivf.get_ex_bits(),
        cur_ivf.get_long_code_device(),
        reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
        cur_ivf.get_ids_device(),
        d_topk_dists,
        d_topk_pids,
        d_query_write_counters,
        num_bits,  // Add num_bits parameter
        num_words  // Add num_words parameter
      );
    }
    else {
      computeInnerProductsWithBitwiseOpt8bitNoEX<<<gridDim, blockDim, shared_mem_size, stream_>>>(
        d_sorted_pairs,
        d_query,
        cur_ivf.get_short_data_device(),  // This is already transposed bit-packed data
        d_cluster_meta,
        d_packed_queries,  // Packed query bit planes
        d_widths,          // Query scaling factors
        cur_ivf.get_short_factors_batch_device(),
        d_G_k1xSumq,
        d_G_kbxSumq,
        get_centroid_distances(),
        topk,
        num_queries,
        nprobe,
        num_pairs,
        cur_ivf.get_num_centroids(),
        D,
        d_topk_threshold_batch,
        15,
        cur_ivf.get_max_cluster_length(),
        cur_ivf.get_ex_bits(),
        cur_ivf.get_long_code_device(),
        reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
        cur_ivf.get_ids_device(),
        d_topk_dists,
        d_topk_pids,
        d_query_write_counters,
        num_bits,  // Add num_bits parameter
        num_words  // Add num_words parameter
      );
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {
    if (cur_ivf.get_ex_bits() != 0) {
      computeInnerProductsWithBitwiseOpt4bit<<<gridDim, blockDim, shared_mem_size, stream_>>>(
        d_sorted_pairs,
        d_query,
        cur_ivf.get_short_data_device(),  // This is already transposed bit-packed data
        d_cluster_meta,
        d_packed_queries,  // Packed query bit planes
        d_widths,          // Query scaling factors
        cur_ivf.get_short_factors_batch_device(),
        d_G_k1xSumq,
        d_G_kbxSumq,
        get_centroid_distances(),
        topk,
        num_queries,
        nprobe,
        num_pairs,
        cur_ivf.get_num_centroids(),
        D,
        d_topk_threshold_batch,
        15,
        cur_ivf.get_max_cluster_length(),
        cur_ivf.get_ex_bits(),
        cur_ivf.get_long_code_device(),
        reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
        cur_ivf.get_ids_device(),
        d_topk_dists,
        d_topk_pids,
        d_query_write_counters,
        num_bits,  // Add num_bits parameter
        num_words  // Add num_words parameter
      );
    } else {
      computeInnerProductsWithBitwiseOpt4bitNoEX<<<gridDim, blockDim, shared_mem_size, stream_>>>(
        d_sorted_pairs,
        d_query,
        cur_ivf.get_short_data_device(),  // This is already transposed bit-packed data
        d_cluster_meta,
        d_packed_queries,  // Packed query bit planes
        d_widths,          // Query scaling factors
        cur_ivf.get_short_factors_batch_device(),
        d_G_k1xSumq,
        d_G_kbxSumq,
        get_centroid_distances(),
        topk,
        num_queries,
        nprobe,
        num_pairs,
        cur_ivf.get_num_centroids(),
        D,
        d_topk_threshold_batch,
        15,
        cur_ivf.get_max_cluster_length(),
        cur_ivf.get_ex_bits(),
        cur_ivf.get_long_code_device(),
        reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device()),
        cur_ivf.get_ids_device(),
        d_topk_dists,
        d_topk_pids,
        d_query_write_counters,
        num_bits,  // Add num_bits parameter
        num_words  // Add num_words parameter
      );
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // Merge results
  mergeClusterTopKFinal(d_topk_dists,
                        d_topk_pids,
                        d_final_dists,
                        d_final_pids,
                        num_queries,
                        nprobe,
                        topk,
                        handle_,
                        /* sorted = */ false);

  // Cleanup
  RAFT_CUDA_TRY(cudaFreeAsync(d_workspace, stream_););

  raft::resource::sync_stream(handle_);
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
