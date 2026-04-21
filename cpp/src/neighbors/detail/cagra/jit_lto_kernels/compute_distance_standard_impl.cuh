/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_standard-impl.cuh"
#include "../compute_distance_standard.hpp"
#include "device_intrinsics.hpp"
#include "device_memory_ops.hpp"

namespace cuvs::neighbors::cagra::detail {

// dist_op is linked from JIT LTO fragments (metric-specific).
template <typename QUERY_T, typename DISTANCE_T>
extern __device__ DISTANCE_T dist_op(QUERY_T a, QUERY_T b);

// Normalization is linked from JIT LTO fragments (cosine or noop).
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
extern __device__ DistanceT apply_normalization_standard(
  DistanceT distance,
  const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
  IndexT dataset_index);

template <typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION auto compute_distance_standard_worker(
  const typename DescriptorT::DATA_T* __restrict__ dataset_ptr,
  uint32_t dim,
  uint32_t query_smem_ptr) -> typename DescriptorT::DISTANCE_T
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using DISTANCE_T                = typename DescriptorT::DISTANCE_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto vlen             = device::get_vlen<LOAD_T, DATA_T>();
  constexpr auto reg_nelem =
    raft::div_rounding_up_unsafe<uint32_t>(kDatasetBlockDim, kTeamSize * vlen);

  DISTANCE_T r = 0;
  for (uint32_t elem_offset = (threadIdx.x % kTeamSize) * vlen; elem_offset < dim;
       elem_offset += kDatasetBlockDim) {
    DATA_T data[reg_nelem][vlen];
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
      device::ldg_cg(reinterpret_cast<LOAD_T&>(data[e]),
                     reinterpret_cast<const LOAD_T*>(dataset_ptr + k));
    }
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
#pragma unroll
      for (uint32_t v = 0; v < vlen; v++) {
        QUERY_T d;
        device::lds(
          d,
          query_smem_ptr +
            sizeof(QUERY_T) * device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(k + v));
        r += dist_op<QUERY_T, DISTANCE_T>(
          d, cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(data[e][v]));
      }
    }
  }
  return r;
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_standard(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  auto distance = compute_distance_standard_worker<DescriptorT>(
    DescriptorT::ptr(args) + (static_cast<std::uint64_t>(DescriptorT::ld(args)) * dataset_index),
    args.dim,
    args.smem_ws_ptr);

  distance =
    apply_normalization_standard<DescriptorT::kTeamSize,
                                 DescriptorT::kDatasetBlockDim,
                                 typename DescriptorT::DATA_T,
                                 typename DescriptorT::INDEX_T,
                                 typename DescriptorT::DISTANCE_T,
                                 typename DescriptorT::QUERY_T>(distance, args, dataset_index);

  return distance;
}

// Unified compute_distance implementation for standard descriptors
// This is instantiated when PQ_BITS=0, PQ_LEN=0, CodebookT=void
// QueryT can be float (for most metrics) or uint8_t (for BitwiseHamming)
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename QueryT>
__device__ DistanceT
compute_distance(const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                 IndexT dataset_index)
{
  // For standard descriptors, PQ_BITS=0, PQ_LEN=0, CodebookT=void
  static_assert(PQ_BITS == 0 && PQ_LEN == 0 && std::is_same_v<CodebookT, void>,
                "Standard descriptor requires PQ_BITS=0, PQ_LEN=0, CodebookT=void");

  using desc_t =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>;
  return compute_distance_standard<desc_t>(args, dataset_index);
}

}  // namespace cuvs::neighbors::cagra::detail
