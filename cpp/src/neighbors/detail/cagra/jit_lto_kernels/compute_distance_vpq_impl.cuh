/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors_device_intrinsics.cuh"
#include "../compute_distance_vpq-impl.cuh"
#include "../compute_distance_vpq.hpp"
#include "device_memory_ops.hpp"
#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::cagra::detail {

template <typename DescriptorT>
_RAFT_DEVICE RAFT_DEVICE_INLINE_FUNCTION auto compute_distance_vpq_worker(
  const uint8_t* __restrict__ dataset_ptr,
  const typename DescriptorT::CODE_BOOK_T* __restrict__ vq_code_book_ptr,
  uint32_t dim,
  uint32_t pq_codebook_ptr) -> typename DescriptorT::DISTANCE_T
{
  using DISTANCE_T               = typename DescriptorT::DISTANCE_T;
  using LOAD_T                   = typename DescriptorT::LOAD_T;
  using QUERY_T                  = typename DescriptorT::QUERY_T;
  using CODE_BOOK_T              = typename DescriptorT::CODE_BOOK_T;
  constexpr auto TeamSize        = DescriptorT::kTeamSize;
  constexpr auto DatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto PQ_BITS         = DescriptorT::kPqBits;
  constexpr auto PQ_LEN          = DescriptorT::kPqLen;

  const uint32_t query_ptr = pq_codebook_ptr + DescriptorT::kSMemCodeBookSizeInBytes;
  static_assert(PQ_BITS == 8, "Only pq_bits == 8 is supported at the moment.");
  constexpr uint32_t vlen = 4;  // **** DO NOT CHANGE ****
  constexpr uint32_t nelem =
    raft::div_rounding_up_unsafe<uint32_t>(DatasetBlockDim / PQ_LEN, TeamSize * vlen);

  constexpr auto kTeamMask = DescriptorT::kTeamSize - 1;
  constexpr auto kTeamVLen = TeamSize * vlen;

  const auto n_subspace = raft::div_rounding_up_unsafe(dim, PQ_LEN);
  const auto laneId     = threadIdx.x & kTeamMask;
  DISTANCE_T norm       = 0;
  for (uint32_t elem_offset = 0; elem_offset * PQ_LEN < dim;
       elem_offset += DatasetBlockDim / PQ_LEN) {
    // Loading PQ codes
    uint32_t pq_codes[nelem];
#pragma unroll
    for (std::uint32_t e = 0; e < nelem; e++) {
      const std::uint32_t k = e * kTeamVLen + elem_offset + laneId * vlen;
      if (k >= n_subspace) break;
      // Loading 4 x 8-bit PQ-codes using 32-bit load ops (from device memory)
      device::ldg_cg(pq_codes[e], reinterpret_cast<const std::uint32_t*>(dataset_ptr + 4 + k));
    }
    //
    if constexpr (PQ_LEN % 2 == 0) {
      // **** Use half2 for distance computation ****
#pragma unroll
      for (std::uint32_t e = 0; e < nelem; e++) {
        const std::uint32_t k = e * kTeamVLen + elem_offset + laneId * vlen;
        if (k >= n_subspace) break;
        // Loading VQ code-book
        half2 vq_vals[PQ_LEN][vlen / 2];
#pragma unroll
        for (std::uint32_t m = 0; m < PQ_LEN; m++) {
          const uint32_t d = (vlen * m) + (PQ_LEN * k);
          if (d >= dim) break;
          device::ldg_ca(vq_vals[m], vq_code_book_ptr + d);
        }
        // Compute distance
        std::uint32_t pq_code = pq_codes[e];
#pragma unroll
        for (std::uint32_t v = 0; v < vlen; v++) {
          if (PQ_LEN * (v + k) >= dim) break;
#pragma unroll
          for (std::uint32_t m = 0; m < PQ_LEN / 2; m++) {
            constexpr auto kQueryBlock = DatasetBlockDim / (vlen * PQ_LEN);
            const std::uint32_t d1     = m + (PQ_LEN / 2) * v;
            const std::uint32_t d =
              d1 * kQueryBlock + elem_offset * (PQ_LEN / 2) + e * TeamSize + laneId;
            half2 q2, c2;
            // Loading query vector from smem
            device::lds(q2, query_ptr + sizeof(half2) * d);
            // Loading PQ code book from smem
            device::lds(c2,
                        pq_codebook_ptr +
                          sizeof(CODE_BOOK_T) * ((1 << PQ_BITS) * 2 * m + (2 * (pq_code & 0xff))));
            // L2 distance
            auto dist = q2 - c2 - reinterpret_cast<half2(&)[PQ_LEN * vlen / 2]>(vq_vals)[d1];
            dist      = dist * dist;
            norm += static_cast<DISTANCE_T>(dist.x + dist.y);
          }
          pq_code >>= 8;
        }
      }
    } else {
      // **** Use float for distance computation ****
#pragma unroll
      for (std::uint32_t e = 0; e < nelem; e++) {
        const std::uint32_t k = e * kTeamVLen + elem_offset + laneId * vlen;
        if (k >= n_subspace) break;
        // Loading VQ code-book
        CODE_BOOK_T vq_vals[PQ_LEN][vlen];
#pragma unroll
        for (std::uint32_t m = 0; m < PQ_LEN; m++) {
          const std::uint32_t d = (vlen * m) + (PQ_LEN * k);
          if (d >= dim) break;
          // Loading 4 x 8/16-bit VQ-values using 32/64-bit load ops (from L2$ or device memory)
          device::ldg_ca(vq_vals[m], vq_code_book_ptr + d);
        }
        // Compute distance
        std::uint32_t pq_code = pq_codes[e];
#pragma unroll
        for (std::uint32_t v = 0; v < vlen; v++) {
          if (PQ_LEN * (v + k) >= dim) break;
          CODE_BOOK_T pq_vals[PQ_LEN];
          device::lds(pq_vals, pq_codebook_ptr + sizeof(CODE_BOOK_T) * PQ_LEN * (pq_code & 0xff));
#pragma unroll
          for (std::uint32_t m = 0; m < PQ_LEN; m++) {
            const std::uint32_t d1 = m + (PQ_LEN * v);
            const std::uint32_t d  = d1 + (PQ_LEN * k);
            DISTANCE_T diff;
            device::lds(diff, query_ptr + sizeof(QUERY_T) * d);
            diff -= static_cast<DISTANCE_T>(pq_vals[m]);
            diff -=
              static_cast<DISTANCE_T>(reinterpret_cast<CODE_BOOK_T(&)[PQ_LEN * vlen]>(vq_vals)[d1]);
            norm += diff * diff;
          }
          pq_code >>= 8;
        }
      }
    }
  }
  return norm;
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_vpq(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  const auto* dataset_ptr =
    DescriptorT::encoded_dataset_ptr(args) +
    (static_cast<std::uint64_t>(DescriptorT::encoded_dataset_dim(args)) * dataset_index);
  uint32_t vq_code;
  device::ldg_cg(vq_code, reinterpret_cast<const std::uint32_t*>(dataset_ptr));
  return compute_distance_vpq_worker<DescriptorT>(
    dataset_ptr /* advance dataset pointer by the size of vq_code */,
    DescriptorT::vq_code_book_ptr(args) + args.dim * vq_code,
    args.dim,
    args.smem_ws_ptr);
}

// Unified compute_distance implementation for VPQ descriptors
// This is instantiated when PQ_BITS>0, PQ_LEN>0, CodebookT=half
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
__device__ DistanceT
compute_distance(const typename dataset_descriptor_base_t<DataT, IndexT, DistanceT>::args_t args,
                 IndexT dataset_index)
{
  // For VPQ descriptors, PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half
  static_assert(
    PQ_BITS > 0 && PQ_LEN > 0 && std::is_same_v<CodebookT, half> && std::is_same_v<QueryT, half>,
    "VPQ descriptor requires PQ_BITS>0, PQ_LEN>0, CodebookT=half, QueryT=half");

  using desc_t = cagra_q_dataset_descriptor_t<TeamSize,
                                              DatasetBlockDim,
                                              PQ_BITS,
                                              PQ_LEN,
                                              CodebookT,
                                              DataT,
                                              IndexT,
                                              DistanceT,
                                              QueryT>;
  return compute_distance_vpq<desc_t>(args, dataset_index);
}

}  // namespace cuvs::neighbors::cagra::detail
