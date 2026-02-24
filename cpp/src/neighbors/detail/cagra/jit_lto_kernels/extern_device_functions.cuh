/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance.hpp"
#include <cuvs/distance/distance.hpp>

namespace cuvs::neighbors::cagra::detail {

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

template <typename SourceIndexT>
extern __device__ bool sample_filter(uint32_t query_id, SourceIndexT node_id, void* filter_data);

}  // namespace cuvs::neighbors::detail
