/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../gpu_index/searcher_gpu_common.cuh"
#include "device_functions.cuh"

#include <raft/util/cuda_dev_essentials.cuh>

#include <cstdint>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Unified kernel template without BlockSort.
// WithEx=true precomputes warp-level IP2 for all cluster vectors; WithEx=false uses only 1-bit
// short codes.
template <bool WithEx>
__device__ void compute_inner_products_with_bitwise_impl(
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

  extern __shared__ __align__(256) char shared_mem_raw[];
  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  float* shared_query = reinterpret_cast<float*>(shared_mem_raw);
  for (size_t i = tid; i < params.D; i += num_threads) {
    shared_query[i] = params.d_query[query_idx * params.D + i];
  }
  __syncthreads();

  if constexpr (WithEx) {
    // Precompute warp-level IP2 for every vector, stored in shared memory
    float* shared_ip2_results     = shared_query + params.D;
    const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;
    const int warp_id             = tid / raft::WarpSize;
    const int lane_id             = tid % raft::WarpSize;
    const int num_warps           = num_threads / raft::WarpSize;

    for (int cand_idx = warp_id; cand_idx < num_vectors_in_cluster; cand_idx += num_warps) {
      size_t global_vec_idx        = cluster_start_index + cand_idx;
      const uint8_t* vec_long_code = params.d_long_code + global_vec_idx * long_code_size;
      float ip2 = compute_ip2_from_long_codes_warp(vec_long_code, shared_query, params.D, lane_id);
      if (lane_id == 0) { shared_ip2_results[cand_idx] = ip2; }
    }
    __syncthreads();
  }

  const size_t short_code_length = params.D / 32;
  float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];

  __shared__ int probe_slot;
  if (tid == 0) {
    probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
  }
  __syncthreads();

  uint32_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

  if constexpr (WithEx) {
    float* shared_ip2_results = shared_query + params.D;
    float q_kbxsumq           = params.d_G_kbxSumq[query_idx];

    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
      size_t vec_idx = vec_base + tid;
      if (vec_idx >= num_vectors_in_cluster) break;

      float exact_ip = compute_bitwise_1bit_ip_for_vec(params.d_short_data,
                                                       shared_query,
                                                       cluster_start_index,
                                                       num_vectors_in_cluster,
                                                       vec_idx,
                                                       short_code_length,
                                                       params.D);

      float ip2             = shared_ip2_results[vec_idx];
      size_t global_vec_idx = cluster_start_index + vec_idx;
      float2 ex_factors     = reinterpret_cast<const float2*>(params.d_ex_factor)[global_vec_idx];
      float f_ex_add        = ex_factors.x;
      float f_ex_rescale    = ex_factors.y;

      params.d_topk_dists[output_offset + vec_idx] =
        f_ex_add + q_g_add +
        f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);
      params.d_topk_pids[output_offset + vec_idx] = params.d_pids[global_vec_idx];
    }
  } else {
    float q_k1xsumq = params.d_G_k1xSumq[query_idx];

    for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
      size_t vec_idx = vec_base + tid;
      if (vec_idx >= num_vectors_in_cluster) break;

      size_t factor_offset = cluster_start_index + vec_idx;
      float3 factors       = reinterpret_cast<const float3*>(params.d_short_factors)[factor_offset];
      float f_add          = factors.x;
      float f_rescale      = factors.y;
      size_t global_vec_idx = cluster_start_index + vec_idx;

      float exact_ip = compute_bitwise_1bit_ip_for_vec(params.d_short_data,
                                                       shared_query,
                                                       cluster_start_index,
                                                       num_vectors_in_cluster,
                                                       vec_idx,
                                                       short_code_length,
                                                       params.D);

      params.d_topk_dists[output_offset + vec_idx] =
        f_add + q_g_add + f_rescale * (exact_ip + q_k1xsumq);
      params.d_topk_pids[output_offset + vec_idx] = params.d_pids[global_vec_idx];
    }
  }
}

}  // namespace cuvs::neighbors::ivf_rabitq::detail
