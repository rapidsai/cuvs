/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors_device_intrinsics.cuh"
#include "../compute_distance_standard-impl.cuh"
#include "../compute_distance_standard.hpp"
#include "device_memory_ops.hpp"

namespace cuvs::neighbors::cagra::detail {

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_standard(
  const DescriptorT* that,
  void* smem_ptr,
  const typename DescriptorT::DATA_T* queries_ptr,
  uint32_t query_id) -> const DescriptorT*
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  using word_type                 = uint32_t;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  auto* r                         = reinterpret_cast<DescriptorT*>(smem_ptr);
  auto* buf                       = reinterpret_cast<QUERY_T*>(r + 1);
  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
    using blob_type           = word_type[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
      r->args.smem_ws_ptr = uint32_t(__cvta_generic_to_shared(buf));
    }
    __syncthreads();
  }

  uint32_t dim        = r->args.dim;
  auto buf_len        = raft::round_up_safe<uint32_t>(dim, kDatasetBlockDim);
  constexpr auto vlen = device::get_vlen<LOAD_T, DATA_T>();
  queries_ptr += dim * query_id;
  for (unsigned i = threadIdx.x; i < buf_len; i += blockDim.x) {
    unsigned j = device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(i);
    if (i < dim) {
      buf[j] = cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(queries_ptr[i]);
    } else {
      buf[j] = 0;
    }
  }
  return r;
}

// Unified setup_workspace implementation for standard descriptors
// This is instantiated when PQ_BITS=0, PQ_LEN=0, CodebookT=void
// Takes const dataset_descriptor_base_t* and reconstructs the derived descriptor inside smem
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
__device__ const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* setup_workspace(
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* desc_ptr,
  void* smem,
  const DataT* queries,
  uint32_t query_id)
{
  // For standard descriptors, PQ_BITS=0, PQ_LEN=0, CodebookT=void
  static_assert(PQ_BITS == 0 && PQ_LEN == 0 && std::is_same_v<CodebookT, void>,
                "Standard descriptor requires PQ_BITS=0, PQ_LEN=0, CodebookT=void");

  // Reconstruct the descriptor pointer from base pointer with QueryT
  using desc_t =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, QueryT>;
  const desc_t* desc = static_cast<const desc_t*>(desc_ptr);

  const desc_t* result = setup_workspace_standard<desc_t>(desc, smem, queries, query_id);
  return static_cast<const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*>(result);
}

}  // namespace cuvs::neighbors::cagra::detail
