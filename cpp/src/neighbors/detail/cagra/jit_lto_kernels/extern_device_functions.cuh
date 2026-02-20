/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// This file contains extern function declarations for JIT LTO
// The actual descriptor type definitions are in -impl.cuh files which are included
// directly by the .cu.in files with CUVS_ENABLE_JIT_LTO defined
// Forward declarations must match the JIT LTO version (no Metric parameter)

#include "../compute_distance.hpp"  // For dataset_descriptor_base_t
#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::cagra::detail {
// Forward declarations matching the JIT LTO version (no Metric parameter, includes QueryT)
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
struct standard_dataset_descriptor_t;

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
struct cagra_q_dataset_descriptor_t;
}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::cagra::detail {

// All extern function declarations are in the cuvs::neighbors::cagra::detail namespace
// so they can be used by all search modes without being beholden to any specific sub-namespace

// Unified setup_workspace and compute_distance extern functions
// These take dataset_descriptor_base_t* and reconstruct the derived descriptor inside
// Standard and VPQ versions are in separate impl headers but use the same function name
// QueryT is needed to reconstruct the descriptor type correctly
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
extern __device__ dataset_descriptor_base_t<DataT, IndexT, DistanceT>* setup_workspace(
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* desc_ptr,
  void* smem,
  const DataT* queries,
  uint32_t query_id);

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
extern __device__ DistanceT
compute_distance(const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
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
