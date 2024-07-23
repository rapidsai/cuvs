/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "search_single_cta_kernel-inl.cuh"
#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

#define instantiate_kernel_selection(TEAM_SIZE, MAX_DATASET_DIM, DATASET_DESC_T, SAMPLE_FILTER_T) \
  template void select_and_run<TEAM_SIZE, MAX_DATASET_DIM, DATASET_DESC_T, SAMPLE_FILTER_T>(      \
    DATASET_DESC_T dataset_desc,                                                                  \
    raft::device_matrix_view<const typename DATASET_DESC_T::INDEX_T, int64_t, raft::row_major>    \
      graph,                                                                                      \
    typename DATASET_DESC_T::INDEX_T* const topk_indices_ptr,                                     \
    typename DATASET_DESC_T::DISTANCE_T* const topk_distances_ptr,                                \
    const typename DATASET_DESC_T::DATA_T* const queries_ptr,                                     \
    const uint32_t num_queries,                                                                   \
    const typename DATASET_DESC_T::INDEX_T* dev_seed_ptr,                                         \
    uint32_t* const num_executed_iterations,                                                      \
    const search_params& ps,                                                                      \
    uint32_t topk,                                                                                \
    uint32_t num_itopk_candidates,                                                                \
    uint32_t block_size,                                                                          \
    uint32_t smem_size,                                                                           \
    int64_t hash_bitlen,                                                                          \
    typename DATASET_DESC_T::INDEX_T* hashmap_ptr,                                                \
    size_t small_hash_bitlen,                                                                     \
    size_t small_hash_reset_interval,                                                             \
    uint32_t num_seeds,                                                                           \
    SAMPLE_FILTER_T sample_filter,                                                                \
    cuvs::distance::DistanceType metric,                                                          \
    cudaStream_t stream);

#define COMMA ,

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
