/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cudart_utils.hpp>

#include "block_sort.cuh"
#include "device_functions.cuh"

namespace cuvs::neighbors::ivf_pq::detail {

/* Manually unrolled loop over a chunk of pq_dataset that fits into one VecT. */
template <typename OutT,
          typename LutT,
          typename VecT,
          bool CheckBounds,
          uint32_t PqBits,
          uint32_t BitsLeft = 0,
          uint32_t Ix       = 0>
__device__ __forceinline__ void compute_chunk(OutT& score /* NOLINT */,
                                              typename VecT::math_t& pq_code,
                                              const VecT& pq_codes,
                                              const LutT*& lut_head,
                                              const LutT*& lut_end)
{
  if constexpr (CheckBounds) {
    if (lut_head >= lut_end) { return; }
  }
  constexpr uint32_t kTotalBits = 8 * sizeof(typename VecT::math_t);
  constexpr uint32_t kPqShift   = 1u << PqBits;
  constexpr uint32_t kPqMask    = kPqShift - 1u;
  if constexpr (BitsLeft >= PqBits) {
    uint8_t code = pq_code & kPqMask;
    pq_code >>= PqBits;
    score += OutT(lut_head[code]);
    lut_head += kPqShift;
    return compute_chunk<OutT, LutT, VecT, CheckBounds, PqBits, BitsLeft - PqBits, Ix>(
      score, pq_code, pq_codes, lut_head, lut_end);
  } else if constexpr (Ix < VecT::Ratio) {
    uint8_t code                = pq_code;
    pq_code                     = pq_codes.val.data[Ix];
    constexpr uint32_t kRemBits = PqBits - BitsLeft;
    constexpr uint32_t kRemMask = (1u << kRemBits) - 1u;
    code |= (pq_code & kRemMask) << BitsLeft;
    pq_code >>= kRemBits;
    score += OutT(lut_head[code]);
    lut_head += kPqShift;
    return compute_chunk<OutT, LutT, VecT, CheckBounds, PqBits, kTotalBits - kRemBits, Ix + 1>(
      score, pq_code, pq_codes, lut_head, lut_end);
  }
}

/* Compute the similarity for one vector in the pq_dataset */
template <typename OutT, typename LutT, typename VecT, uint32_t PqBits>
__device__ auto compute_score(uint32_t pq_dim,
                              const typename VecT::io_t* pq_head,
                              const LutT* lut_scores,
                              OutT early_stop_limit) -> OutT
{
  constexpr uint32_t kChunkSize = sizeof(VecT) * 8u / PqBits;
  auto lut_head                 = lut_scores;
  auto lut_end                  = lut_scores + (pq_dim << PqBits);
  VecT pq_codes;
  OutT score{0};
  for (; pq_dim >= kChunkSize; pq_dim -= kChunkSize) {
    *pq_codes.vectorized_data() = *pq_head;
    pq_head += kIndexGroupSize;
    typename VecT::math_t pq_code = 0;
    compute_chunk<OutT, LutT, VecT, false, PqBits>(score, pq_code, pq_codes, lut_head, lut_end);
    // Early stop when it makes sense (otherwise early_stop_limit is kDummy/infinity).
    if (score >= early_stop_limit) { return score; }
  }
  if (pq_dim > 0) {
    *pq_codes.vectorized_data()   = *pq_head;
    typename VecT::math_t pq_code = 0;
    compute_chunk<OutT, LutT, VecT, true, PqBits>(score, pq_code, pq_codes, lut_head, lut_end);
  }
  return score;
}

template <typename OutT, typename LutT, int Capacity, uint32_t PqBits>
__device__ void compute_distances_impl(const uint32_t* chunk_indices,
                                       float* query_kths,
                                       uint32_t n_probes,
                                       uint32_t probe_ix,
                                       uint32_t query_ix,
                                       uint32_t max_samples,
                                       distance::DistanceType metric,
                                       uint32_t topk,
                                       uint8_t* lut_end,
                                       uint32_t pq_dim,
                                       const uint8_t* const* pq_dataset,
                                       uint32_t label,
                                       uint32_t queries_offset,
                                       OutT* out_scores,
                                       uint32_t* out_indices,
                                       LutT* lut_scores,
                                       uint8_t* smem_buf,
                                       filtering::ivf_filter_dev sample_filter)
{
  constexpr bool kManageLocalTopK = Capacity > 0;

  // Define helper types for efficient access to the pq_dataset, which is stored in an interleaved
  // format. The chunks of PQ data are stored in kIndexGroupVecLen-bytes-long chunks, interleaved
  // in groups of kIndexGroupSize elems (which is normally equal to the warp size) for the fastest
  // possible access by thread warps.
  //
  // Consider one record in the pq_dataset is `pq_dim * pq_bits`-bit-long.
  // Assuming `kIndexGroupVecLen = 16`, one chunk of data read by a thread at once is 128-bits.
  // Then, such a chunk contains `chunk_size = 128 / pq_bits` record elements, and the record
  // consists of `ceildiv(pq_dim, chunk_size)` chunks. The chunks are interleaved in groups of 32,
  // so that the warp can achieve the best coalesced read throughput.
  using group_align  = raft::Pow2<kIndexGroupSize>;
  using vec_align    = raft::Pow2<kIndexGroupVecLen>;
  using local_topk_t = block_sort_t<Capacity, OutT, uint32_t>;
  using op_t         = uint32_t;
  using vec_t        = raft::TxN_t<op_t, kIndexGroupVecLen / sizeof(op_t)>;

  uint32_t sample_offset = 0;
  if (probe_ix > 0) { sample_offset = chunk_indices[probe_ix - 1]; }
  uint32_t n_samples            = chunk_indices[probe_ix] - sample_offset;
  uint32_t n_samples_aligned    = group_align::roundUp(n_samples);
  constexpr uint32_t kChunkSize = (kIndexGroupVecLen * 8u) / PqBits;
  uint32_t pq_line_width = raft::div_rounding_up_unsafe(pq_dim, kChunkSize) * kIndexGroupVecLen;
  auto pq_thread_data    = pq_dataset[label] + group_align::roundDown(threadIdx.x) * pq_line_width +
                        group_align::mod(threadIdx.x) * vec_align::Value;
  pq_line_width *= blockDim.x;

  constexpr OutT kDummy = raft::upper_bound<OutT>();
  OutT query_kth        = kDummy;
  if constexpr (kManageLocalTopK) { query_kth = OutT(query_kths[query_ix]); }
  OutT early_stop_limit = kDummy;
  switch (metric) {
    // If the metric is non-negative, we can use the query_kth approximation as an early stop
    // threshold to skip some iterations when computing the score. Add such metrics here.
    case distance::DistanceType::L2SqrtExpanded:
    case distance::DistanceType::L2Expanded: {
      early_stop_limit = query_kth;
    } break;
    default: break;
  }

  // Ensure lut_scores is written by all threads before using it in ivfpq-compute-score
  __threadfence_block();
  __syncthreads();
  local_topk_t block_topk(topk, lut_end, query_kth);

  // Compute a distance for each sample
  for (uint32_t i = threadIdx.x; i < n_samples_aligned;
       i += blockDim.x, pq_thread_data += pq_line_width) {
    OutT score = kDummy;
    bool valid = i < n_samples;
    // Check bounds and that the sample is acceptable for the query
    if (valid && sample_filter(queries_offset + query_ix, label, i)) {
      score = compute_score<OutT, LutT, vec_t, PqBits>(
        pq_dim, reinterpret_cast<const vec_t::io_t*>(pq_thread_data), lut_scores, early_stop_limit);
      if (metric == distance::DistanceType::CosineExpanded) { score = OutT(1) + score; }
    }
    if constexpr (kManageLocalTopK) {
      block_topk.add(score, sample_offset + i);
    } else {
      if (valid) { out_scores[sample_offset + i] = score; }
    }
  }
  if constexpr (kManageLocalTopK) {
    // sync threads before the topk merging operation, because we reuse smem_buf
    __syncthreads();
    block_topk.done(smem_buf);
    block_topk.store(out_scores, out_indices);
    if (threadIdx.x == 0) { atomicMin(query_kths + query_ix, float(out_scores[topk - 1])); }
  } else {
    // fill in the rest of the out_scores with dummy values
    if (probe_ix + 1 == n_probes) {
      for (uint32_t i = threadIdx.x + sample_offset + n_samples; i < max_samples; i += blockDim.x) {
        out_scores[i] = kDummy;
      }
    }
  }
}

}  // namespace cuvs::neighbors::ivf_pq::detail
