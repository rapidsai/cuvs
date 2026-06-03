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
template <int EX_BITS>
__device__ uint32_t extract_code(const uint8_t* codes, size_t d);

__device__ float compute_ip2_from_long_codes_warp(
  const uint8_t* vec_long_code, const float* shared_query, size_t D, size_t EX_BITS, int lane_id);

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

__device__ float compute_bitwise_1bit_ip_for_vec(const uint32_t* d_short_data,
                                                 const float* shared_query,
                                                 size_t cluster_start_index,
                                                 size_t num_vectors_in_cluster,
                                                 size_t vec_idx,
                                                 size_t short_code_length,
                                                 size_t D);

__device__ void update_threshold_atomicmin(const float* d_topk_dists,
                                           const float* d_threshold,
                                           uint32_t query_idx,
                                           uint32_t topk,
                                           uint32_t nprobe,
                                           int probe_slot,
                                           float threshold,
                                           int tid);

__device__ int32_t compute_bitwise_quantized_ip_for_vec(const uint32_t* d_short_data,
                                                        const uint32_t* shared_packed_query,
                                                        size_t cluster_start_index,
                                                        size_t num_vectors_in_cluster,
                                                        size_t vec_idx,
                                                        uint32_t num_words);

}  // namespace cuvs::neighbors::ivf_rabitq::detail
