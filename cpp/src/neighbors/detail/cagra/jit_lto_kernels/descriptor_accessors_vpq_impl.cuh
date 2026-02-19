/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_vpq-impl.cuh"

namespace cuvs::neighbors::cagra::detail {

// Descriptor accessor fragments for VPQ descriptors
// These take void* and reconstruct the descriptor pointer, then return the member
// Same function names as standard versions - planner links the right fragment at runtime

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ uint32_t get_dim(void* desc_ptr)
{
  using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                    DatasetBlockDim,
                                                    PQ_BITS,
                                                    PQ_LEN,
                                                    CodebookT,
                                                    DataT,
                                                    IndexT,
                                                    DistanceT>;
  const desc_t* desc = reinterpret_cast<const desc_t*>(desc_ptr);
  return desc->args.dim;
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ IndexT get_size(void* desc_ptr)
{
  using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                    DatasetBlockDim,
                                                    PQ_BITS,
                                                    PQ_LEN,
                                                    CodebookT,
                                                    DataT,
                                                    IndexT,
                                                    DistanceT>;
  const desc_t* desc = reinterpret_cast<const desc_t*>(desc_ptr);
  return desc->size;
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ uint32_t get_team_size_bitshift(void* desc_ptr)
{
  using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                    DatasetBlockDim,
                                                    PQ_BITS,
                                                    PQ_LEN,
                                                    CodebookT,
                                                    DataT,
                                                    IndexT,
                                                    DistanceT>;
  const desc_t* desc = reinterpret_cast<const desc_t*>(desc_ptr);
  // Use team_size_bitshift() which works for both global and shared memory descriptors
  // team_size_bitshift_from_smem() only works when descriptor is in shared memory
  return desc->team_size_bitshift();
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ uint32_t get_team_size_bitshift_from_smem(void* desc_ptr)
{
  using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                    DatasetBlockDim,
                                                    PQ_BITS,
                                                    PQ_LEN,
                                                    CodebookT,
                                                    DataT,
                                                    IndexT,
                                                    DistanceT>;
  const desc_t* desc = reinterpret_cast<const desc_t*>(desc_ptr);
  // Use team_size_bitshift_from_smem() which is optimized for shared memory access
  return desc->team_size_bitshift_from_smem();
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t get_args(
  void* desc_ptr)
{
  using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                    DatasetBlockDim,
                                                    PQ_BITS,
                                                    PQ_LEN,
                                                    CodebookT,
                                                    DataT,
                                                    IndexT,
                                                    DistanceT>;
  const desc_t* desc = reinterpret_cast<const desc_t*>(desc_ptr);
  return desc->args.load();
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__device__ uint32_t get_smem_ws_size_in_bytes(void* desc_ptr, uint32_t dim)
{
  using desc_t = cagra_q_dataset_descriptor_t<TeamSize,
                                              DatasetBlockDim,
                                              PQ_BITS,
                                              PQ_LEN,
                                              CodebookT,
                                              DataT,
                                              IndexT,
                                              DistanceT>;
  return desc_t::get_smem_ws_size_in_bytes(dim);
}

}  // namespace cuvs::neighbors::cagra::detail
