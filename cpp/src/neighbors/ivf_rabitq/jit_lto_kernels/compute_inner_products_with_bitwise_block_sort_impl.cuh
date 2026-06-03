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

// Unified kernel template using BlockSort.
// NumBits=4 or 8; WithEx=true adds warp-level IP2 refinement with long codes.
template <int NumBits, bool WithEx>
__device__ void compute_inner_products_with_bitwise_block_sort_impl(
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
        if (cand_idx < num_candidates) {
          int vec_idx                    = shared_candidate_indices[cand_idx];
          float exact_ip                 = compute_bitwise_1bit_ip_for_vec(params.d_short_data,
                                                           shared_query,
                                                           cluster_start_index,
                                                           num_vectors_in_cluster,
                                                           vec_idx,
                                                           short_code_length,
                                                           params.D);
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

          float ip2 = compute_ip2_from_long_codes_warp(
            vec_long_code, shared_query, params.D, params.ex_bits, lane_id);
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
        if (cand_idx < num_candidates) {
          int vec_idx          = shared_candidate_indices[cand_idx];
          size_t factor_offset = cluster_start_index + vec_idx;
          float3 factors  = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
          float f_add     = factors.x;
          float f_rescale = factors.y;
          size_t global_vec_idx = cluster_start_index + vec_idx;

          float exact_ip = compute_bitwise_1bit_ip_for_vec(params.d_short_data,
                                                           shared_query,
                                                           cluster_start_index,
                                                           num_vectors_in_cluster,
                                                           vec_idx,
                                                           short_code_length,
                                                           params.D);
          final_dist     = f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
          final_pid      = (uint32_t)params.d_pids[global_vec_idx];
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

}  // namespace cuvs::neighbors::ivf_rabitq::detail
