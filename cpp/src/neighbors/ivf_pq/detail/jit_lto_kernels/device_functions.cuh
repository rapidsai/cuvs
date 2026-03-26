/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <neighbors/sample_filter.cuh>

namespace cuvs::neighbors::ivf_pq::detail {

template <typename LutT>
__device__ void prepare_lut(uint8_t* smem_buf,
                            uint32_t pq_dim,
                            LutT*& lut_scores,
                            uint8_t*& lut_end);

template <typename OutT>
__device__ void store_calculated_distances(OutT* _out_scores,
                                           uint32_t* _out_indices,
                                           uint32_t probe_ix,
                                           uint32_t n_probes,
                                           uint32_t query_ix,
                                           uint32_t topk,
                                           uint32_t max_samples,
                                           OutT*& out_scores,
                                           uint32_t*& out_indices);

__device__ void precompute_base_diff(cuvs::distance::DistanceType metric,
                                     uint32_t dim,
                                     uint8_t* lut_end,
                                     const float* query,
                                     const float* cluster_center);

template <typename LutT>
__device__ void create_lut(uint32_t pq_dim,
                           uint32_t pq_len,
                           uint32_t label,
                           const float* pq_centers,
                           codebook_gen codebook_kind,
                           cuvs::distance::DistanceType metric,
                           LutT* lut_scores,
                           uint8_t* lut_end,
                           const float* query,
                           const float* cluster_center);

template <typename OutT, typename LutT>
__device__ void compute_distances(const uint32_t* chunk_indices,
                                  float* query_kths,
                                  uint32_t n_probes,
                                  uint32_t probe_ix,
                                  uint32_t query_ix,
                                  uint32_t max_samples,
                                  distance::DistanceType metric,
                                  uint32_t topk,
                                  uint8_t* lut_end,
                                  uint32_t pq_dim,
                                  const uint8_t* const* pq_dataset,
                                  uint32_t label,
                                  uint32_t queries_offset,
                                  OutT* out_scores,
                                  uint32_t* out_indices,
                                  LutT* lut_scores,
                                  uint8_t* smem_buf,
                                  filtering::ivf_filter_dev sample_filter);

}  // namespace cuvs::neighbors::ivf_pq::detail
