/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "ivf_pq_codepacking.cuh"
#include <cstdint>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/pow2_utils.cuh>
#include <variant>

namespace cuvs::neighbors::ivf_pq::detail {

using cuvs::neighbors::ivf_pq::kIndexGroupSize;
using cuvs::neighbors::ivf_pq::kIndexGroupVecLen;

/**
 * A consumer for the `run_on_vector` that just flattens PQ codes
 * into a tightly packed matrix. That is, the codes are not expanded to one code-per-byte.
 */
template <uint32_t PqBits>
struct unpack_contiguous {
  uint8_t* codes;
  uint32_t code_size;

  /**
   * Create a callable to be passed to `run_on_vector`.
   *
   * @param[in] codes flat compressed PQ codes
   */
  __host__ __device__ inline unpack_contiguous(uint8_t* codes, uint32_t pq_dim)
    : codes{codes}, code_size{raft::ceildiv<uint32_t>(pq_dim * PqBits, 8)}
  {
  }

  /**  Write j-th component (code) of the i-th vector into the output array. */
  __host__ __device__ inline void operator()(uint8_t code, uint32_t i, uint32_t j)
  {
    bitfield_view_t<PqBits> code_view{codes + i * code_size};
    code_view[j] = code;
  }
};

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void unpack_contiguous_list_data_kernel(
  uint8_t* out_codes,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    in_list_data,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  run_on_list<PqBits>(
    in_list_data, offset_or_indices, n_rows, pq_dim, unpack_contiguous<PqBits>(out_codes, pq_dim));
}

/**
 * Unpack flat PQ codes from an existing list by the given offset.
 *
 * @param[out] codes flat compressed PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] list_data the packed ivf::list data.
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void unpack_contiguous_list_data_impl(
  uint8_t* codes,
  raft::device_mdspan<const uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return unpack_contiguous_list_data_kernel<kBlockSize, 4>;
      case 5: return unpack_contiguous_list_data_kernel<kBlockSize, 5>;
      case 6: return unpack_contiguous_list_data_kernel<kBlockSize, 6>;
      case 7: return unpack_contiguous_list_data_kernel<kBlockSize, 7>;
      case 8: return unpack_contiguous_list_data_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream>>>(codes, list_data, n_rows, pq_dim, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * A producer for the `write_vector` reads tightly packed flat codes. That is,
 * the codes are not expanded to one code-per-byte.
 */
template <uint32_t PqBits>
struct pack_contiguous {
  const uint8_t* codes;
  uint32_t code_size;

  /**
   * Create a callable to be passed to `write_vector`.
   *
   * @param[in] codes flat compressed PQ codes
   */
  __host__ __device__ inline pack_contiguous(const uint8_t* codes, uint32_t pq_dim)
    : codes{codes}, code_size{raft::ceildiv<uint32_t>(pq_dim * PqBits, 8)}
  {
  }

  /** Read j-th component (code) of the i-th vector from the source. */
  __host__ __device__ inline auto operator()(uint32_t i, uint32_t j) -> uint8_t
  {
    bitfield_view_t<PqBits> code_view{const_cast<uint8_t*>(codes + i * code_size)};
    return uint8_t(code_view[j]);
  }
};

/**
 * Pack a vector by copying chunks directly.
 */
template <uint32_t PqBits>
__device__ inline void pack_vector_chunks(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    out_list_data,
  uint32_t out_ix,
  const uint8_t* codes,
  uint32_t pq_dim)
{
  using group_align         = raft::Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(out_ix);
  const uint32_t ingroup_ix = group_align::mod(out_ix);

  constexpr uint32_t chunk_bytes     = sizeof(pq_vec_t);
  constexpr uint32_t codes_per_chunk = (chunk_bytes * 8u) / PqBits;
  uint32_t n_chunks                  = raft::ceildiv<uint32_t>(pq_dim, codes_per_chunk);

  for (uint32_t i = 0; i < n_chunks; i++) {
    pq_vec_t chunk;
    if (i < n_chunks - 1 || (pq_dim % codes_per_chunk) == 0) {
      chunk = *reinterpret_cast<const pq_vec_t*>(codes + i * chunk_bytes);
    } else {
      chunk                   = pq_vec_t{};
      uint32_t occupied_bytes = raft::ceildiv<uint32_t>((pq_dim % codes_per_chunk) * PqBits, 8);
      auto* chunk_bytes_ptr   = reinterpret_cast<uint8_t*>(&chunk);
      for (uint32_t j = 0; j < occupied_bytes; j++) {
        chunk_bytes_ptr[j] = codes[i * chunk_bytes + j];
      }
    }
    *reinterpret_cast<pq_vec_t*>(&out_list_data(group_ix, i, ingroup_ix, 0)) = chunk;
  }
}

/**
 * Pack a vector by extracting each code (for unaligned cases: pq_bits = 5, 6, 7)
 */
template <uint32_t PqBits>
__device__ inline void pack_vector(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    out_list_data,
  uint32_t out_ix,
  const uint8_t* codes,
  uint32_t pq_dim)
{
  using group_align         = raft::Pow2<kIndexGroupSize>;
  const uint32_t group_ix   = group_align::div(out_ix);
  const uint32_t ingroup_ix = group_align::mod(out_ix);

  pq_vec_t code_chunk = pq_vec_t{};
  bitfield_view_t<PqBits> src_view{const_cast<uint8_t*>(codes)};
  bitfield_view_t<PqBits> dst_view{reinterpret_cast<uint8_t*>(&code_chunk)};
  constexpr uint32_t kChunkSize = (sizeof(pq_vec_t) * 8u) / PqBits;

  for (uint32_t j = 0, i = 0; j < pq_dim; i++) {
    for (uint32_t k = 0; k < kChunkSize && j < pq_dim; k++, j++) {
      dst_view[k] = src_view[j];
    }
    *reinterpret_cast<pq_vec_t*>(&out_list_data(group_ix, i, ingroup_ix, 0)) = code_chunk;
    if (j < pq_dim) code_chunk = pq_vec_t{};
  }
}

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void pack_contiguous_list_data_kernel(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  write_list<PqBits, 1>(
    list_data, offset_or_indices, n_rows, pq_dim, pack_contiguous<PqBits>(codes, pq_dim));
}

template <uint32_t BlockSize, uint32_t PqBits>
__launch_bounds__(BlockSize) static __global__ void pack_list_chunks_kernel(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices)
{
  uint32_t ix = threadIdx.x + blockDim.x * blockIdx.x;
  if (ix >= n_rows) return;

  const uint32_t dst_ix = std::holds_alternative<uint32_t>(offset_or_indices)
                            ? std::get<uint32_t>(offset_or_indices) + ix
                            : std::get<const uint32_t*>(offset_or_indices)[ix];

  const uint32_t code_size = raft::ceildiv<uint32_t>(pq_dim * PqBits, 8);
  const uint8_t* src_codes = codes + ix * code_size;

  if constexpr (PqBits == 4 || PqBits == 8) {
    // aligned case: direct chunk copies
    pack_vector_chunks<PqBits>(list_data, dst_ix, src_codes, pq_dim);
  } else {
    // unaligned case: extract each code
    pack_vector<PqBits>(list_data, dst_ix, src_codes, pq_dim);
  }
}

/**
 * Write flat PQ codes into an existing list by the given offset.
 *
 * NB: no memory allocation happens here; the list must fit the data (offset + n_rows).
 *
 * @param[out] list_data the packed ivf::list data.
 * @param[in] codes flat compressed PQ codes [n_rows, ceildiv(pq_dim * pq_bits, 8)]
 * @param[in] offset_or_indices how many records in the list to skip or the exact indices.
 * @param[in] pq_bits codebook size (1 << pq_bits)
 * @param[in] stream
 */
inline void pack_contiguous_list_data_impl(
  raft::device_mdspan<uint8_t, list_spec<uint32_t, uint32_t>::list_extents, raft::row_major>
    list_data,
  const uint8_t* codes,
  uint32_t n_rows,
  uint32_t pq_dim,
  std::variant<uint32_t, const uint32_t*> offset_or_indices,
  uint32_t pq_bits,
  rmm::cuda_stream_view stream)
{
  if (n_rows == 0) { return; }

  constexpr uint32_t kBlockSize = 256;
  dim3 blocks(raft::div_rounding_up_safe<uint32_t>(n_rows, kBlockSize), 1, 1);
  dim3 threads(kBlockSize, 1, 1);
  auto kernel = [pq_bits]() {
    switch (pq_bits) {
      case 4: return pack_list_chunks_kernel<kBlockSize, 4>;
      case 5: return pack_list_chunks_kernel<kBlockSize, 5>;
      case 6: return pack_list_chunks_kernel<kBlockSize, 6>;
      case 7: return pack_list_chunks_kernel<kBlockSize, 7>;
      case 8: return pack_list_chunks_kernel<kBlockSize, 8>;
      default: RAFT_FAIL("Invalid pq_bits (%u), the value must be within [4, 8]", pq_bits);
    }
  }();
  kernel<<<blocks, threads, 0, stream>>>(list_data, codes, n_rows, pq_dim, offset_or_indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}
}  // namespace cuvs::neighbors::ivf_pq::detail
