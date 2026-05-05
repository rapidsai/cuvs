/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Created by Stardust on 4/14/25.
//

// This file implements `SearcherGPU::SearchClusterQueryPairs`.
#include "../../detail/smem_utils.cuh"
#include "../../ivf_flat/detail/jit_lto_kernels/interleaved_scan_impl.cuh"
#include "../utils/memory.hpp"
#include "../utils/searcher_gpu_utils.hpp"
#include "searcher_gpu.cuh"
#include "searcher_gpu_common.cuh"

#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select_k.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <optional>
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

void SearcherGPU::AllocateSearcherSpace(size_t num_centroids, size_t num_queries)
{
  centroid_distances_ =
    raft::make_device_vector<float, int64_t>(handle_, num_queries * num_centroids);
  c_norms_ = raft::make_device_vector<float, int64_t>(handle_, num_centroids);
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
  // Launch precompute kernel
  dim3 gridDim(num_queries, 1, 1);
  dim3 blockDim(256, 1, 1);  // Can tune this

  precomputeAllLUTs<<<gridDim, blockDim, 0, stream>>>(d_query, d_lut_for_queries, num_queries, D);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// compute distances and write to output
// Unified non-BlockSort LUT kernel.
// WithEx=true: IP2 precomputed with long codes for all cluster vectors before the distance loop.
// WithEx=false: direct 1-bit LUT distance, no IP2 step.
template <bool WithEx>
__global__ void computeInnerProductsWithLUT(const ComputeInnerProductsKernelParams params)
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

// Unified BlockSort LUT kernel.
// WithEx=true: LUT filter with error bound → IP2 refinement with long codes.
// WithEx=false: LUT filter → direct sort with PIDs stored separately from vec indices.
template <bool WithEx>
__global__ void computeInnerProductsWithLUTBlockSort(const ComputeInnerProductsKernelParams params)
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
  const size_t lut_bytes          = lut_per_query_size * sizeof(float);

  extern __shared__ __align__(256) char shared_mem_raw[];
  float* shared_lut = reinterpret_cast<float*>(shared_mem_raw);

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  float* query_lut = params.d_lut_for_queries_float + query_idx * lut_per_query_size;
  for (size_t i = tid; i < lut_per_query_size; i += num_threads) {
    shared_lut[i] = query_lut[i];
  }
  __syncthreads();

  __shared__ int num_candidates;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_k1xsumq = params.d_G_k1xSumq[query_idx];
  float threshold = params.d_threshold[query_idx];

  if (tid == 0) { num_candidates = 0; }
  __syncthreads();

  const size_t short_code_length  = params.D / 32;
  const size_t chunks_per_uint32  = 32 / BITS_PER_CHUNK;
  const int vectors_per_iteration = num_threads;

  if constexpr (WithEx) {
    float q_g_error = sqrtf(q_g_add);

    // Shared memory layout after LUT: [cand_dists][cand_ips][cand_indices][queue_buffer]
    float* shared_candidate_dists = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
    float* shared_candidate_ips   = shared_candidate_dists + params.max_candidates_per_pair;
    int* shared_candidate_indices =
      reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);
    int* shared_buffer = shared_candidate_indices + params.max_candidates_per_pair;

    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster;
         vec_base += vectors_per_iteration) {
      size_t vec_idx = vec_base + tid;

      float local_low_dist = INFINITY;
      float local_ip       = 0.0f;
      bool is_candidate    = false;

      if (vec_idx < num_vectors_in_cluster) {
        size_t factor_offset = cluster_start_index + vec_idx;
        float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
        float f_add     = factors.x;
        float f_rescale = factors.y;
        float f_error   = factors.z;

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

        float est_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);
        float low_dist = est_dist - f_error * q_g_error;

        if (low_dist < threshold) {
          is_candidate   = true;
          local_low_dist = est_dist;
          local_ip       = ip;
        }
      }
      __syncwarp();

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

    int final_num_candidates = min(num_candidates, (int)params.max_candidates_per_pair);

    __syncthreads();
    if (final_num_candidates > 0) {
      using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
        flat_block_sort<kMaxTopKBlockSort, true, T, IdxT>::type;
      block_sort_t queue(params.topk);

      float q_kbxsumq             = params.d_G_kbxSumq[query_idx];
      const size_t long_code_size = (params.D * params.ex_bits + 7) / 8;

      float* shared_query = reinterpret_cast<float*>(shared_lut);
      for (size_t i = tid; i < params.D; i += num_threads) {
        shared_query[i] = params.d_query[query_idx * params.D + i];
      }
      __syncthreads();

      float* shared_ip2_results = shared_candidate_dists;

      const int warp_id   = tid / raft::WarpSize;
      const int lane_id   = tid % raft::WarpSize;
      const int num_warps = num_threads / raft::WarpSize;

      for (int cand_idx = warp_id; cand_idx < final_num_candidates; cand_idx += num_warps) {
        int local_vec_idx            = shared_candidate_indices[cand_idx];
        size_t global_vec_idx        = cluster_start_index + local_vec_idx;
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

      const int adds_per_thread = (final_num_candidates + num_threads - 1) / num_threads;
      for (int round = 0; round < adds_per_thread; round++) {
        int cand_idx = tid + round * num_threads;

        float ex_dist;
        uint32_t pid;

        if (cand_idx < final_num_candidates) {
          float ip              = shared_candidate_ips[cand_idx];
          float ip2             = shared_ip2_results[cand_idx];
          int local_vec_idx     = shared_candidate_indices[cand_idx];
          size_t global_vec_idx = cluster_start_index + local_vec_idx;

          float2 ex_factors  = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
          float f_ex_add     = ex_factors.x;
          float f_ex_rescale = ex_factors.y;

          ex_dist = f_ex_add + q_g_add +
                    f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);
          pid = (uint32_t)params.d_pids[global_vec_idx];
        } else {
          ex_dist = INFINITY;
          pid     = 0;
        }
        queue.add(ex_dist, pid);
      }
      __syncthreads();

      queue.done((uint8_t*)shared_buffer);

      __shared__ int probe_slot;
      if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
      __syncthreads();

      if (probe_slot >= params.nprobe) { return; }

      size_t output_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
      queue.store(params.d_topk_dists + output_offset,
                  (uint32_t*)(params.d_topk_pids + output_offset));

      if (final_num_candidates >= params.topk) {
        float max_topk_dist;
        if (tid == 0) {
          max_topk_dist = -INFINITY;
          size_t output_offset =
            query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
          for (size_t i = 0; i < params.topk; i++) {
            float dist = params.d_topk_dists[output_offset + i];
            if (dist > 0 && dist > max_topk_dist && dist < INFINITY) { max_topk_dist = dist; }
          }
        }
        __syncthreads();

        if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
          int* threshold_ptr = (int*)(params.d_threshold + query_idx);
          int new_val        = __float_as_int(max_topk_dist);
          atomicMin(threshold_ptr, new_val);
        }
      }
    }
  } else {
    // Shared memory layout after LUT: [cand_ips][cand_pids]
    float* shared_candidate_ips = reinterpret_cast<float*>(shared_mem_raw + lut_bytes);
    int* shared_candidate_pids =
      reinterpret_cast<int*>(shared_candidate_ips + params.max_candidates_per_pair);

    float final_1bit_dist;
    PID final_1bit_pid;

    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster;
         vec_base += vectors_per_iteration) {
      size_t vec_idx = vec_base + tid;

      bool is_candidate = false;

      if (vec_idx < num_vectors_in_cluster) {
        size_t factor_offset = cluster_start_index + vec_idx;
        float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
        float f_add     = factors.x;
        float f_rescale = factors.y;

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

        final_1bit_dist = f_add + q_g_add + f_rescale * (ip + q_k1xsumq);

        if (final_1bit_dist < threshold) {
          is_candidate   = true;
          final_1bit_pid = params.d_pids[cluster_start_index + vec_idx];
        }
      }
      __syncwarp();

      if (is_candidate) {
        int candidate_slot = atomicAdd(&num_candidates, 1);
        if (candidate_slot < params.max_candidates_per_pair) {
          shared_candidate_ips[candidate_slot]  = final_1bit_dist;
          shared_candidate_pids[candidate_slot] = final_1bit_pid;
        }
      }
    }
    __syncthreads();

    __syncthreads();
    if (num_candidates > 0) {
      __shared__ int probe_slot;
      {
        using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
          flat_block_sort<kMaxTopKBlockSort, true, T, IdxT>::type;
        block_sort_t queue(params.topk);

        const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

        for (int c = 0; c < candidates_per_thread; ++c) {
          int cand_idx = tid + c * num_threads;

          if (cand_idx < num_candidates && cand_idx < params.max_candidates_per_pair) {
            final_1bit_dist = shared_candidate_ips[cand_idx];
            final_1bit_pid  = shared_candidate_pids[cand_idx];
          } else {
            final_1bit_dist = INFINITY;
            final_1bit_pid  = 0;
          }
          queue.add(final_1bit_dist, final_1bit_pid);
        }
        __syncthreads();

        queue.done((uint8_t*)shared_lut);

        if (tid == 0) { probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1); }
        __syncthreads();

        if (probe_slot >= params.nprobe) { return; }

        uint32_t output_offset =
          query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;
        queue.store(params.d_topk_dists + output_offset,
                    (uint32_t*)(params.d_topk_pids + output_offset));
      }

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
          int* threshold_ptr = (int*)(params.d_threshold + query_idx);
          int new_val        = __float_as_int(max_topk_dist);
          atomicMin(threshold_ptr, new_val);
        }
      }
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
                                          float* d_final_dists,
                                          PID* d_final_pids)
{
  // First allocate space for LUT
  size_t lut_size =
    num_queries * (cur_ivf.get_num_padded_dim() / BITS_PER_CHUNK) * LUT_SIZE * sizeof(float);
  // each line's space is (cur_ivf.get_num_padded_dim() / BITS_PER_CHUNK) * LUT_SIZE *
  // sizeof(float);
  rmm::device_uvector<float> d_lut_for_queries(lut_size / sizeof(float), stream_);
  thrust::fill(thrust::cuda::par.on(stream_),
               d_lut_for_queries.data(),
               d_lut_for_queries.data() + d_lut_for_queries.size(),
               -std::numeric_limits<float>::infinity());
  // precompute LUTS
  launchPrecomputeLUTs(d_query, d_lut_for_queries.data(), num_queries, D, stream_);

  // check if the inner products kernel should use block sort to keep a top-k priority queue vs.
  // outputting distances from all vectors in probed clusters
  const bool use_block_sort{topk <= kMaxTopKBlockSort};

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

  rmm::device_uvector<int> d_query_write_counters(num_queries, stream_);
  thrust::fill(thrust::cuda::par.on(stream_),
               d_query_write_counters.data(),
               d_query_write_counters.data() + num_queries,
               0);

  rmm::device_uvector<float> d_topk_threshold_batch(use_block_sort ? num_queries : 0, stream_);
  if (use_block_sort) {
    thrust::fill(thrust::cuda::par.on(stream_),
                 d_topk_threshold_batch.data(),
                 d_topk_threshold_batch.data() + num_queries,
                 std::numeric_limits<float>::infinity());
  }
  // Then launch kernel for computation
  size_t num_pairs = num_queries * nprobe;
  uint32_t gridDim{static_cast<uint32_t>(num_pairs)};
  uint32_t blockDim{256};
  size_t num_chunks = D / BITS_PER_CHUNK;
  size_t candidate_storage =
    max_cluster_size * (use_block_sort ? (2 * sizeof(float) /* ip, ip2 */ + sizeof(int) /* idx */)
                                       : sizeof(float) /* ip2 */);
  const int queue_buffer_smem_bytes =
    use_block_sort ? raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<T, IdxT>(
                       blockDim / raft::WarpSize, kMaxTopKBlockSort)
                   : 0;

  ComputeInnerProductsKernelParams kernelParams;
  kernelParams.d_sorted_pairs          = d_sorted_pairs;
  kernelParams.d_query                 = d_query;
  kernelParams.d_short_data            = cur_ivf.get_short_data_device();
  kernelParams.d_cluster_meta          = d_cluster_meta;
  kernelParams.d_lut_for_queries_float = d_lut_for_queries.data();
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
  kernelParams.d_threshold             = d_topk_threshold_batch.data();
  kernelParams.max_candidates_per_pair = max_cluster_size;
  kernelParams.max_candidates_per_query =
    use_block_sort ? 0 /* unused */ : max_probed_vectors_count.value();
  kernelParams.ex_bits      = cur_ivf.get_ex_bits();
  kernelParams.d_long_code  = cur_ivf.get_long_code_device();
  kernelParams.d_ex_factor  = reinterpret_cast<const float*>(cur_ivf.get_ex_factor_device());
  kernelParams.d_pids       = cur_ivf.get_ids_device();
  kernelParams.d_topk_dists = d_topk_dists.data_handle();
  kernelParams.d_topk_pids  = d_topk_pids.data_handle();
  kernelParams.d_query_write_counters = d_query_write_counters.data();

  if (cur_ivf.get_ex_bits() != 0) {
    auto kernel = use_block_sort ? computeInnerProductsWithLUTBlockSort<true>
                                 : computeInnerProductsWithLUT<true>;
    size_t shared_mem_size =
      num_chunks * LUT_SIZE * sizeof(float) + candidate_storage + queue_buffer_smem_bytes;
    auto const& kernel_launcher = [&](auto const& kernel) -> void {
      kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
    };
    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
      kernel, shared_mem_size, kernel_launcher);
  } else {
    auto kernel = use_block_sort ? computeInnerProductsWithLUTBlockSort<false>
                                 : computeInnerProductsWithLUT<false>;
    size_t shared_mem_size =
      max(num_chunks * LUT_SIZE * sizeof(float) +
            (use_block_sort ? max_cluster_size * (sizeof(float) + sizeof(int)) : 0),
          (size_t)queue_buffer_smem_bytes);
    auto const& kernel_launcher = [&](auto const& kernel) -> void {
      kernel<<<gridDim, blockDim, shared_mem_size, stream_>>>(kernelParams);
    };
    cuvs::neighbors::detail::safely_launch_kernel_with_smem_size(
      kernel, shared_mem_size, kernel_launcher);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // merge results from different blocks
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
