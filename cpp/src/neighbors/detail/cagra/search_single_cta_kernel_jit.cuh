/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_distance-ext.cuh"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "utils.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>

#include <cfloat>
#include <cstdint>
#include <cuda_fp16.h>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Enum to distinguish between descriptor types
enum class DescriptorType { Standard, VPQ };

// These extern device functions are linked at runtime using JIT-LTO.
// They are templated on descriptor parameters (not DescriptorT) and create
// descriptor instances internally.

// Standard descriptor extern functions
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ uint32_t setup_workspace_standard(void* smem,
                                                    const DataT* queries,
                                                    uint32_t query_id,
                                                    const DataT* dataset_ptr,
                                                    IndexT dataset_size,
                                                    uint32_t dim,
                                                    uint32_t ld,
                                                    const DistanceT* dataset_norms = nullptr);

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ DistanceT compute_distance_standard(const DataT* dataset_ptr,
                                                      uint32_t smem_ws_ptr,
                                                      IndexT dataset_index,
                                                      uint32_t dim,
                                                      uint32_t ld,
                                                      uint32_t team_size_bitshift,
                                                      const DistanceT* dataset_norms = nullptr);

// VPQ descriptor extern functions
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ uint32_t setup_workspace_vpq(void* smem,
                                               const DataT* queries,
                                               uint32_t query_id,
                                               const uint8_t* encoded_dataset_ptr,
                                               uint32_t encoded_dataset_dim,
                                               const CodebookT* vq_code_book_ptr,
                                               const CodebookT* pq_code_book_ptr,
                                               IndexT dataset_size,
                                               uint32_t dim);

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ DistanceT compute_distance_vpq(const uint8_t* encoded_dataset_ptr,
                                                 uint32_t smem_ws_ptr,
                                                 IndexT dataset_index,
                                                 uint32_t encoded_dataset_dim,
                                                 const CodebookT* vq_code_book_ptr,
                                                 const CodebookT* pq_code_book_ptr,
                                                 uint32_t team_size_bitshift);

// Sample filter extern function
template <typename SourceIndexT>
extern __device__ bool sample_filter(uint32_t query_id, SourceIndexT node_id);

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search

// Include the implementation
#include "search_single_cta_kernel_jit-inl.cuh"
