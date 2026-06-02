/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Suppress the inline `extract_code` definition in searcher_gpu_common.cuh;
// device_functions.cuh provides the extern declaration and the body lives in
// the extract_code JIT-LTO fragment.
#define IVF_RABITQ_JIT_LTO_FRAGMENT
#include "../../detail/smem_utils.cuh"
#include "../../ivf_flat/detail/jit_lto_kernels/interleaved_scan_impl.cuh"
#include "../gpu_index/searcher_gpu_common.cuh"
#include "device_functions.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

#include <cstdint>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Unified BlockSort LUT kernel body.
// WithEx=true: LUT filter with error bound → IP2 refinement with long codes.
// WithEx=false: LUT filter → direct sort with PIDs stored separately from vec indices.
template <bool WithEx>
__device__ void compute_inner_products_with_lut_block_sort_impl(
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

        float ip = compute_lut_ip_for_vec<float>(params.d_short_data,
                                                 shared_lut,
                                                 cluster_start_index,
                                                 num_vectors_in_cluster,
                                                 vec_idx,
                                                 short_code_length);

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

        float ip2 = compute_ip2_from_long_codes_warp(
          vec_long_code, shared_query, params.D, params.ex_bits, lane_id);
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

        float ip = compute_lut_ip_for_vec<float>(params.d_short_data,
                                                 shared_lut,
                                                 cluster_start_index,
                                                 num_vectors_in_cluster,
                                                 vec_idx,
                                                 short_code_length);

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
    if (num_candidates > 0) {
      __shared__ int probe_slot;
      {
        using block_sort_t = typename cuvs::neighbors::ivf_flat::detail::
          flat_block_sort<kMaxTopKBlockSort, true, T, IdxT>::type;
        block_sort_t queue(params.topk);

        const int candidates_per_thread = (num_candidates + num_threads - 1) / num_threads;

        for (int c = 0; c < candidates_per_thread; ++c) {
          int cand_idx = tid + c * num_threads;

          if (cand_idx < num_candidates) {
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

}  // namespace cuvs::neighbors::ivf_rabitq::detail
