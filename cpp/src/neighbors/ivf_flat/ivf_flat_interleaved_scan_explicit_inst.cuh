/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/ann_utils.cuh"
#ifdef CUVS_ENABLE_JIT_LTO
#include "ivf_flat_interleaved_scan_jit.cuh"
#else
#include "ivf_flat_interleaved_scan.cuh"
#endif
#include <cstdint>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#define CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(T, IdxT, SampleFilterT)                        \
  template void                                                                            \
  ivfflat_interleaved_scan<T,                                                              \
                           typename cuvs::spatial::knn::detail::utils::config<T>::value_t, \
                           IdxT,                                                           \
                           SampleFilterT>(const index<T, IdxT>& index,                     \
                                          const search_params& params,                     \
                                          const T* queries,                                \
                                          const uint32_t* coarse_query_results,            \
                                          const uint32_t n_queries,                        \
                                          const uint32_t queries_offset,                   \
                                          const cuvs::distance::DistanceType metric,       \
                                          const uint32_t n_probes,                         \
                                          const uint32_t k,                                \
                                          const uint32_t max_samples,                      \
                                          const uint32_t* chunk_indices,                   \
                                          const bool select_min,                           \
                                          SampleFilterT sample_filter,                     \
                                          uint32_t* neighbors,                             \
                                          float* distances,                                \
                                          uint32_t& grid_dim_x,                            \
                                          rmm::cuda_stream_view stream);

#define COMMA ,
