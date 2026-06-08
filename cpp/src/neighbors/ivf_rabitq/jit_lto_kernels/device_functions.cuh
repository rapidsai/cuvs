/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_rabitq::detail {

// Cross-fragment device function declarations. Definitions live in their own
// JIT-LTO fragments and are resolved at nvJitLink time when a consumer planner
// adds them via add_*_device_function().
__device__ uint32_t extract_code(const uint8_t* codes, size_t d);

__device__ float compute_ip2_from_long_codes_warp(const uint8_t* vec_long_code,
                                                  const float* shared_query,
                                                  size_t D,
                                                  int lane_id);

// Primary template; explicit instantiations live in the
// compute_lut_ip_for_vec_kernel.cu.in fragment, one per @lut_dtype@ substitution
// (float and __half today). Consumers call the specialization matching the
// element type of their `shared_lut`.
template <typename LutT>
__device__ float compute_lut_ip_for_vec(const uint32_t* d_short_data,
                                        const LutT* shared_lut,
                                        size_t cluster_start_index,
                                        size_t num_vectors_in_cluster,
                                        size_t vec_idx,
                                        size_t short_code_length);

__inline__ __device__ float compute_bitwise_1bit_ip_for_vec(const uint32_t* d_short_data,
                                                            const float* shared_query,
                                                            size_t cluster_start_index,
                                                            size_t num_vectors_in_cluster,
                                                            size_t vec_idx,
                                                            size_t short_code_length,
                                                            size_t D)
{
  float exact_ip = 0.0f;
  for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
    size_t short_code_offset =
      cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
    uint32_t short_code_chunk = d_short_data[short_code_offset];
#pragma unroll 8
    for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
      size_t dim = uint32_idx * 32 + bit_idx;
      if (dim < D) {
        if ((short_code_chunk >> (31 - bit_idx)) & 0x1) { exact_ip += shared_query[dim]; }
      }
    }
  }
  return exact_ip;
}

// Note: contains __syncthreads(); must be called by all threads of the block.
__inline__ __device__ void update_threshold_atomicmin(const float* d_topk_dists,
                                                      const float* d_threshold,
                                                      uint32_t query_idx,
                                                      uint32_t topk,
                                                      uint32_t nprobe,
                                                      int probe_slot,
                                                      float threshold,
                                                      int tid)
{
  float max_topk_dist;
  if (tid == 0) {
    max_topk_dist          = -INFINITY;
    uint32_t output_offset = query_idx * (topk * nprobe) + probe_slot * topk;
    for (uint32_t i = 0; i < topk; i++) {
      float dist = d_topk_dists[output_offset + i];
      if (dist > 0 && dist > max_topk_dist && dist < INFINITY) { max_topk_dist = dist; }
    }
  }
  __syncthreads();
  if (tid == 0 && max_topk_dist > 0 && max_topk_dist < threshold) {
    int* threshold_ptr = (int*)(d_threshold + query_idx);
    atomicMin(threshold_ptr, __float_as_int(max_topk_dist));
  }
}

__device__ int32_t compute_bitwise_quantized_ip_for_vec(const uint32_t* d_short_data,
                                                        const uint32_t* shared_packed_query,
                                                        size_t cluster_start_index,
                                                        size_t num_vectors_in_cluster,
                                                        size_t vec_idx,
                                                        uint32_t num_words);

}  // namespace cuvs::neighbors::ivf_rabitq::detail
