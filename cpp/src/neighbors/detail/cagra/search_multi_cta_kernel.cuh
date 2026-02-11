/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_distance-ext.cuh"

#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra::detail::multi_cta_search {

template <typename DataT,
          typename index_t,
          typename distance_t,
          typename SourceIndexT,
          typename SampleFilterT>
void select_and_run(const dataset_descriptor_host<DataT, index_t, distance_t>& dataset_desc,
                    raft::device_matrix_view<const index_t, int64_t, raft::row_major> graph,
                    const SourceIndexT* source_indices_ptr,
                    index_t* topk_indices_ptr,       // [num_queries, topk]
                    distance_t* topk_distances_ptr,  // [num_queries, topk]
                    const DataT* queries_ptr,        // [num_queries, dataset_dim]
                    uint32_t num_queries,
                    const index_t* dev_seed_ptr,        // [num_queries, num_seeds]
                    uint32_t* num_executed_iterations,  // [num_queries,]
                    const search_params& ps,
                    uint32_t topk,
                    // multi_cta_search (params struct)
                    uint32_t block_size,  //
                    uint32_t result_buffer_size,
                    uint32_t smem_size,
                    uint32_t visited_hash_bitlen,
                    int64_t traversed_hash_bitlen,
                    index_t* traversed_hashmap_ptr,
                    uint32_t num_cta_per_query,
                    uint32_t num_seeds,
                    SampleFilterT sample_filter,
                    cudaStream_t stream);

}
