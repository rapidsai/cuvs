/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../compute_distance_vpq-impl.cuh"
#include "../compute_distance_vpq.hpp"
#include "device_intrinsics.hpp"
#include "device_memory_ops.hpp"

namespace cuvs::neighbors::cagra::detail {

template <auto Block, auto Stride, typename T>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto transpose(T x) -> T
{
  auto i = x % Block;
  auto j = x / Block;
  auto k = i % Stride;
  auto l = i / Stride;
  return j * Block + k * (Block / Stride) + l;
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_vpq(const DescriptorT* that,
                                                   void* smem_ptr,
                                                   const typename DescriptorT::DATA_T* queries_ptr,
                                                   uint32_t query_id) -> const DescriptorT*
{
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  using CODE_BOOK_T               = typename DescriptorT::CODE_BOOK_T;
  using word_type                 = uint32_t;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto PQ_BITS          = DescriptorT::kPqBits;
  constexpr auto PQ_LEN           = DescriptorT::kPqLen;

  auto* r = reinterpret_cast<DescriptorT*>(smem_ptr);

  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
    using blob_type           = word_type[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }

    auto codebook_buf = uint32_t(__cvta_generic_to_shared(r + 1));
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
      r->args.smem_ws_ptr = codebook_buf;
    }
    __syncthreads();

    // Copy PQ table
    for (unsigned i = threadIdx.x * 2; i < (1 << PQ_BITS) * PQ_LEN; i += blockDim.x * 2) {
      half2 buf2;
      buf2.x = r->pq_code_book_ptr()[i];
      buf2.y = r->pq_code_book_ptr()[i + 1];

      // Change the order of PQ code book array to reduce the
      // frequency of bank conflicts.
      constexpr auto num_elements_per_bank  = 4 / utils::size_of<CODE_BOOK_T>();
      constexpr auto num_banks_per_subspace = PQ_LEN / num_elements_per_bank;
      const auto j                          = i / num_elements_per_bank;
      const auto smem_index =
        (j / num_banks_per_subspace) + (j % num_banks_per_subspace) * (1 << PQ_BITS);

      device::sts(codebook_buf + smem_index * sizeof(half2), buf2);
    }
  }

  uint32_t dim = r->args.dim;
  queries_ptr += dim * query_id;

  constexpr cuvs::spatial::knn::detail::utils::mapping<QUERY_T> mapping{};
  auto smem_query_ptr =
    reinterpret_cast<QUERY_T*>(reinterpret_cast<uint8_t*>(smem_ptr) + sizeof(DescriptorT) +
                               DescriptorT::kSMemCodeBookSizeInBytes);
  for (unsigned i = threadIdx.x * 2; i < dim; i += blockDim.x * 2) {
    half2 buf2{0, 0};
    if (i < dim) { buf2.x = mapping(queries_ptr[i]); }
    if (i + 1 < dim) { buf2.y = mapping(queries_ptr[i + 1]); }
    if constexpr ((PQ_BITS == 8) && (PQ_LEN % 2 == 0)) {
      // Transpose the queries buffer to avoid bank conflicts in compute_distance.
      constexpr uint32_t vlen = 4;  // **** DO NOT CHANGE ****
      constexpr auto kStride  = vlen * PQ_LEN / 2;
      reinterpret_cast<half2*>(smem_query_ptr)[transpose<kDatasetBlockDim / 2, kStride>(i / 2)] =
        buf2;
    } else {
      (reinterpret_cast<half2*>(smem_query_ptr + i))[0] = buf2;
    }
  }

  return r;
}

// Unified setup_workspace implementation for VPQ descriptors
// This is instantiated when PQ_BITS>0, PQ_LEN>0, CodebookT=half
// Takes const dataset_descriptor_base_t* and reconstructs the derived descriptor inside smem
// QueryT is always half for VPQ
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
  // For VPQ descriptors, PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half
  static_assert(
    PQ_BITS > 0 && PQ_LEN > 0 && std::is_same_v<CodebookT, half> && std::is_same_v<QueryT, half>,
    "VPQ descriptor requires PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half");

  using desc_t       = cagra_q_dataset_descriptor_t<TeamSize,
                                                    DatasetBlockDim,
                                                    PQ_BITS,
                                                    PQ_LEN,
                                                    CodebookT,
                                                    DataT,
                                                    IndexT,
                                                    DistanceT,
                                                    QueryT>;
  const desc_t* desc = static_cast<const desc_t*>(desc_ptr);

  const desc_t* result = setup_workspace_vpq<desc_t>(desc, smem, queries, query_id);
  return static_cast<const dataset_descriptor_base_t<DataT, IndexT, DistanceT>*>(result);
}

}  // namespace cuvs::neighbors::cagra::detail
