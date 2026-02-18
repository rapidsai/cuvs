/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance.hpp"
#include <cuvs/distance/distance.hpp>

// Forward declarations of descriptor types (full definitions are in -impl.cuh files)
// This file is only included by JIT kernel headers which are included by .cu.in files
// The .cu.in files include the -impl.cuh files directly for full type definitions
namespace cuvs::neighbors::cagra::detail {
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct standard_dataset_descriptor_t;

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct cagra_q_dataset_descriptor_t;
}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::cagra::detail {

// All extern function declarations are in the cuvs::neighbors::cagra::detail namespace
// so they can be used by all search modes without being beholden to any specific sub-namespace

// Standard descriptor extern functions
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ const
  standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>*
  setup_workspace_standard(const standard_dataset_descriptor_t<Metric,
                                                               TeamSize,
                                                               DatasetBlockDim,
                                                               DataT,
                                                               IndexT,
                                                               DistanceT>* desc,
                           void* smem,
                           const DataT* queries,
                           uint32_t query_id);

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ DistanceT compute_distance_standard(
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index);

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
extern __device__ const cagra_q_dataset_descriptor_t<Metric,
                                                     TeamSize,
                                                     DatasetBlockDim,
                                                     PQ_BITS,
                                                     PQ_LEN,
                                                     CodebookT,
                                                     DataT,
                                                     IndexT,
                                                     DistanceT>*
setup_workspace_vpq(const cagra_q_dataset_descriptor_t<Metric,
                                                       TeamSize,
                                                       DatasetBlockDim,
                                                       PQ_BITS,
                                                       PQ_LEN,
                                                       CodebookT,
                                                       DataT,
                                                       IndexT,
                                                       DistanceT>* desc,
                    void* smem,
                    const DataT* queries,
                    uint32_t query_id);

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
extern __device__ DistanceT compute_distance_vpq(
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index);

}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::detail {

// Sample filter extern function - linked separately via JIT LTO
// Takes 3 params: query_id, node_id, and filter_data (void* pointer to filter-specific data)
// For none_filter: filter_data can be nullptr
// For bitset_filter: filter_data points to bitset_filter_data_t struct
template <typename SourceIndexT>
extern __device__ bool sample_filter(uint32_t query_id, SourceIndexT node_id, void* filter_data);

}  // namespace cuvs::neighbors::detail
