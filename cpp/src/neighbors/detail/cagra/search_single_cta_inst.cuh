/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "search_single_cta_kernel-inl.cuh"
#include <cuvs/neighbors/common.hpp>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

#define instantiate_kernel_selection(DataT, IndexT, DistanceT, SampleFilterT) \
  template void select_and_run<DataT, IndexT, DistanceT, SampleFilterT>(      \
    const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,    \
    raft::device_matrix_view<const IndexT, int64_t, raft::row_major> graph,   \
    uintptr_t topk_indices_ptr,                                               \
    DistanceT* topk_distances_ptr,                                            \
    const DataT* queries_ptr,                                                 \
    uint32_t num_queries,                                                     \
    const IndexT* dev_seed_ptr,                                               \
    uint32_t* num_executed_iterations,                                        \
    const search_params& ps,                                                  \
    uint32_t topk,                                                            \
    uint32_t num_itopk_candidates,                                            \
    uint32_t block_size,                                                      \
    uint32_t smem_size,                                                       \
    int64_t hash_bitlen,                                                      \
    IndexT* hashmap_ptr,                                                      \
    size_t small_hash_bitlen,                                                 \
    size_t small_hash_reset_interval,                                         \
    uint32_t num_seeds,                                                       \
    SampleFilterT sample_filter,                                              \
    cudaStream_t stream);

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
