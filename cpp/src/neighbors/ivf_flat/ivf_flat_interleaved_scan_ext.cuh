/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

#include "../detail/ann_utils.cuh"
#include <cuvs/neighbors/common.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/raft_explicit.hpp>

namespace cuvs::neighbors::ivf_flat::detail {
template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void ivfflat_interleaved_scan(const index<T, IdxT>& index,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const uint32_t queries_offset,
                              const cuvs::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const uint32_t max_samples,
                              const uint32_t* chunk_indices,
                              const bool select_min,
                              IvfSampleFilterT sample_filter,
                              uint32_t* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream) RAFT_EXPLICIT;

#define CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(T, IdxT, SampleFilterT)                        \
  extern template void                                                                     \
  ivfflat_interleaved_scan<T,                                                              \
                           typename cuvs::spatial::knn::detail::utils::config<T>::value_t, \
                           IdxT,                                                           \
                           SampleFilterT>(const index<T, IdxT>& index,                     \
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

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(float, int64_t, cuvs::neighbors::filtering::none_sample_filter);

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(half, int64_t, cuvs::neighbors::filtering::none_sample_filter);

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(int8_t,
                                    int64_t,
                                    cuvs::neighbors::filtering::none_sample_filter);

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(uint8_t,
                                    int64_t,
                                    cuvs::neighbors::filtering::none_sample_filter);

#define COMMA ,
CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(
  float, int64_t, cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>);

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(
  half, int64_t, cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>);

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(
  int8_t, int64_t, cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>);

CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN(
  uint8_t, int64_t, cuvs::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>);
#undef COMMA
#undef CUVS_INST_IVF_FLAT_INTERLEAVED_SCAN

}  // namespace cuvs::neighbors::ivf_flat::detail
