/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "compute_distance.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodeBookT,
          typename DataT,
          typename IndexT,
          typename DistanceT = float>
struct cagra_q_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type   = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using CODE_BOOK_T = CodeBookT;
  using LOAD_T      = device::LOAD_128BIT_T;
  using QUERY_T     = half;
  using base_type::dim;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::ws_handle;

  static_assert(std::is_same_v<CODE_BOOK_T, half>,
                "Only CODE_BOOK_T = "
                "`half` is supported "
                "now");

  const std::uint8_t* encoded_dataset_ptr;
  const CODE_BOOK_T* vq_code_book_ptr;
  const CODE_BOOK_T* pq_code_book_ptr;
  std::uint32_t encoded_dataset_dim;
  std::uint32_t n_subspace;
  float vq_scale;
  float pq_scale;

  static constexpr std::uint32_t kSMemCodeBookSizeInBytes =
    (1 << PQ_BITS) * PQ_LEN * utils::size_of<CODE_BOOK_T>();

  _RAFT_HOST_DEVICE cagra_q_dataset_descriptor_t(const std::uint8_t* encoded_dataset_ptr,
                                                 std::uint32_t encoded_dataset_dim,
                                                 std::uint32_t n_subspace,
                                                 const CODE_BOOK_T* vq_code_book_ptr,
                                                 float vq_scale,
                                                 const CODE_BOOK_T* pq_code_book_ptr,
                                                 float pq_scale,
                                                 std::size_t size,
                                                 std::uint32_t dim)
    : base_type(size, dim),
      encoded_dataset_ptr(encoded_dataset_ptr),
      encoded_dataset_dim(encoded_dataset_dim),
      n_subspace(n_subspace),
      vq_code_book_ptr(vq_code_book_ptr),
      vq_scale(vq_scale),
      pq_code_book_ptr(pq_code_book_ptr),
      pq_scale(pq_scale)
  {
    base_type::template assert_struct_size<sizeof(*this)>();
  }
  _RAFT_HOST_DEVICE [[nodiscard]] auto team_size() const -> uint32_t { return TeamSize; }

  _RAFT_HOST_DEVICE [[nodiscard]] auto smem_ws_size_in_bytes() const -> uint32_t
  {
    /* SMEM workspace layout:
      1. Codebook (kSMemCodeBookSizeInBytes bytes)
      2. Queries (smem_query_buffer_length elems)
    */
    return kSMemCodeBookSizeInBytes +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }

  _RAFT_DEVICE [[nodiscard]] auto set_smem_ws(void* smem_ptr) const -> ws_handle
  {
    auto codebook_buf = reinterpret_cast<half2*>(smem_ptr);

    // Copy PQ table
    for (unsigned i = threadIdx.x * 2; i < (1 << PQ_BITS) * PQ_LEN; i += blockDim.x * 2) {
      half2 buf2;
      buf2.x = pq_code_book_ptr[i];
      buf2.y = pq_code_book_ptr[i + 1];

      // Change the order of PQ code book array to reduce the
      // frequency of bank conflicts.
      constexpr auto num_elements_per_bank  = 4 / utils::size_of<CODE_BOOK_T>();
      constexpr auto num_banks_per_subspace = PQ_LEN / num_elements_per_bank;
      const auto j                          = i / num_elements_per_bank;
      const auto smem_index =
        (j / num_banks_per_subspace) + (j % num_banks_per_subspace) * (1 << PQ_BITS);
      codebook_buf[smem_index] = buf2;
    }
    return reinterpret_cast<ws_handle>(smem_ptr);
  }

  _RAFT_DEVICE void copy_query(ws_handle smem_workspace, const DATA_T* query_ptr) const
  {
    constexpr cuvs::spatial::knn::detail::utils::mapping<half> mapping{};
    auto smem_query_ptr = smem_query_buffer(smem_workspace);
    for (unsigned i = threadIdx.x * 2; i < dim; i += blockDim.x * 2) {
      half2 buf2{0, 0};
      if (i < dim) { buf2.x = mapping(query_ptr[i]); }
      if (i + 1 < dim) { buf2.y = mapping(query_ptr[i + 1]); }
      if ((PQ_BITS == 8) && (PQ_LEN % 2 == 0)) {
        // Use swizzling in the condition to reduce bank conflicts in shared
        // memory, which are likely to occur when pq_code_book_dim is large.
        ((half2*)smem_query_ptr)[device::swizzling<std::uint32_t, DatasetBlockDim / 2>(i / 2)] =
          buf2;
      } else {
        (reinterpret_cast<half2*>(smem_query_ptr + i))[0] = buf2;
      }
    }
  }

  _RAFT_DEVICE auto compute_distance(ws_handle smem_workspace,
                                     INDEX_T dataset_index,
                                     cuvs::distance::DistanceType metric,
                                     bool valid) const -> DISTANCE_T
  {
    switch (metric) {
      case cuvs::distance::DistanceType::L2Expanded:
        return compute_similarity<cuvs::distance::DistanceType::L2Expanded>(
          smem_workspace, dataset_index, valid);
      case cuvs::distance::DistanceType::InnerProduct:
        return compute_similarity<cuvs::distance::DistanceType::InnerProduct>(
          smem_workspace, dataset_index, valid);
      default: return 0;
    }
  }

  template <cuvs::distance::DistanceType METRIC>
  RAFT_DEVICE_INLINE_FUNCTION DISTANCE_T compute_similarity(ws_handle smem_workspace,
                                                            const INDEX_T node_id,
                                                            const bool valid) const
  {
    auto codebook_ptr = smem_pq_code_book_ptr(smem_workspace);
    auto query_ptr    = smem_query_buffer(smem_workspace);
    float norm        = 0;
    if (valid) {
      const unsigned lane_id = threadIdx.x % TeamSize;
      const uint32_t vq_code = *(reinterpret_cast<const std::uint32_t*>(
        encoded_dataset_ptr + (static_cast<std::uint64_t>(encoded_dataset_dim) * node_id)));
      if (PQ_BITS == 8) {
        for (uint32_t elem_offset = 0; elem_offset < dim; elem_offset += DatasetBlockDim) {
          constexpr unsigned vlen = 4;  // **** DO NOT CHANGE ****
          constexpr unsigned nelem =
            raft::div_rounding_up_unsafe<unsigned>(DatasetBlockDim / PQ_LEN, TeamSize * vlen);
          // Loading PQ codes
          uint32_t pq_codes[nelem];
#pragma unroll
          for (std::uint32_t e = 0; e < nelem; e++) {
            const std::uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset / PQ_LEN;
            if (k >= n_subspace) break;
            // Loading 4 x 8-bit PQ-codes using 32-bit load ops (from device memory)
            pq_codes[e] = *(reinterpret_cast<const std::uint32_t*>(
              encoded_dataset_ptr + (static_cast<std::uint64_t>(encoded_dataset_dim) * node_id) +
              4 + k));
          }
          //
          if constexpr (PQ_LEN % 2 == 0) {
            // **** Use half2 for distance computation ****
            half2 norm2{0, 0};
#pragma unroll
            for (std::uint32_t e = 0; e < nelem; e++) {
              const std::uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset / PQ_LEN;
              if (k >= n_subspace) break;
              // Loading VQ code-book
              raft::TxN_t<half2, vlen / 2> vq_vals[PQ_LEN];
#pragma unroll
              for (std::uint32_t m = 0; m < PQ_LEN; m += 1) {
                const uint32_t d = (vlen * m) + (PQ_LEN * k);
                if (d >= dim) break;
                vq_vals[m].load(
                  reinterpret_cast<const half2*>(vq_code_book_ptr + d + (dim * vq_code)), 0);
              }
              // Compute distance
              std::uint32_t pq_code = pq_codes[e];
#pragma unroll
              for (std::uint32_t v = 0; v < vlen; v++) {
                if (PQ_LEN * (v + k) >= dim) break;
#pragma unroll
                for (std::uint32_t m = 0; m < PQ_LEN; m += 2) {
                  const std::uint32_t d1 = m + (PQ_LEN * v);
                  const std::uint32_t d  = d1 + (PQ_LEN * k);
                  // Loading query vector in smem
                  half2 diff2 = (reinterpret_cast<const half2*>(
                    query_ptr))[device::swizzling<std::uint32_t, DatasetBlockDim / 2>(d / 2)];
                  // Loading PQ code book in smem
                  diff2 -= *(reinterpret_cast<half2*>(codebook_ptr + (1 << PQ_BITS) * 2 * (m / 2) +
                                                      (2 * (pq_code & 0xff))));
                  diff2 -= vq_vals[d1 / vlen].val.data[(d1 % vlen) / 2];
                  norm2 += diff2 * diff2;
                }
                pq_code >>= 8;
              }
            }
            norm += static_cast<float>(norm2.x + norm2.y);
          } else {
            // **** Use float for distance computation ****
#pragma unroll
            for (std::uint32_t e = 0; e < nelem; e++) {
              const std::uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset / PQ_LEN;
              if (k >= n_subspace) break;
              // Loading VQ code-book
              raft::TxN_t<CODE_BOOK_T, vlen> vq_vals[PQ_LEN];
#pragma unroll
              for (std::uint32_t m = 0; m < PQ_LEN; m++) {
                const std::uint32_t d = (vlen * m) + (PQ_LEN * k);
                if (d >= dim) break;
                // Loading 4 x 8/16-bit VQ-values using 32/64-bit load ops (from L2$ or device
                // memory)
                vq_vals[m].load(
                  reinterpret_cast<const half2*>(vq_code_book_ptr + d + (dim * vq_code)), 0);
              }
              // Compute distance
              std::uint32_t pq_code = pq_codes[e];
#pragma unroll
              for (std::uint32_t v = 0; v < vlen; v++) {
                if (PQ_LEN * (v + k) >= dim) break;
                raft::TxN_t<CODE_BOOK_T, PQ_LEN> pq_vals;
                pq_vals.load(
                  reinterpret_cast<const half2*>(codebook_ptr + PQ_LEN * (pq_code & 0xff)),
                  0);  // (from L1$ or smem)
#pragma unroll
                for (std::uint32_t m = 0; m < PQ_LEN; m++) {
                  const std::uint32_t d1 = m + (PQ_LEN * v);
                  const std::uint32_t d  = d1 + (PQ_LEN * k);
                  // if (d >= dataset_dim) break;
                  DISTANCE_T diff = query_ptr[d];  // (from smem)
                  diff -= pq_scale * static_cast<float>(pq_vals.data[m]);
                  diff -= vq_scale * static_cast<float>(vq_vals[d1 / vlen].val.data[d1 % vlen]);
                  norm += diff * diff;
                }
                pq_code >>= 8;
              }
            }
          }
        }
      }
    }
    for (uint32_t offset = TeamSize / 2; offset > 0; offset >>= 1) {
      norm += __shfl_xor_sync(0xffffffff, norm, offset);
    }
    return norm;
  }

 private:
  RAFT_DEVICE_INLINE_FUNCTION constexpr auto smem_pq_code_book_ptr(ws_handle smem_workspace) const
    -> CODE_BOOK_T*
  {
    return reinterpret_cast<CODE_BOOK_T*>(smem_workspace);
  }

  RAFT_DEVICE_INLINE_FUNCTION constexpr auto smem_query_buffer(ws_handle smem_workspace) const
    -> QUERY_T*
  {
    return reinterpret_cast<QUERY_T*>(reinterpret_cast<uint8_t*>(smem_workspace) +
                                      kSMemCodeBookSizeInBytes);
  }
};

extern template struct standard_dataset_descriptor_t<8, 128, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 2, half, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 4, half, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 2, half, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 4, half, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 2, half, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 4, half, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 2, half, float, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 4, half, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 2, half, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 4, half, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 2, half, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 4, half, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 2, half, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 4, half, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 2, half, half, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 4, half, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 2, half, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 4, half, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 2, half, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 4, half, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 2, half, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 4, half, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 2, half, int8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 4, half, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 2, half, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 4, half, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 2, half, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 4, half, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 2, half, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 4, half, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 2, half, uint8_t, uint32_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 4, half, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 2, half, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 4, half, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 2, half, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 4, half, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 2, half, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 4, half, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 2, half, float, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 4, half, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 2, half, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<8, 128, 8, 4, half, half, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 2, half, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<16, 256, 8, 4, half, half, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 2, half, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 512, 8, 4, half, half, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 2, half, half, uint64_t, float>;
extern template struct cagra_q_dataset_descriptor_t<32, 1024, 8, 4, half, half, uint64_t, float>;

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodeBookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__launch_bounds__(1, 1) __global__
  void vpq_dataset_descriptor_init_kernel(dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
                                          const std::uint8_t* encoded_dataset_ptr,
                                          std::uint32_t encoded_dataset_dim,
                                          std::uint32_t n_subspace,
                                          const CodeBookT* vq_code_book_ptr,
                                          float vq_scale,
                                          const CodeBookT* pq_code_book_ptr,
                                          float pq_scale,
                                          std::size_t size,
                                          std::uint32_t dim)
{
  new (out) cagra_q_dataset_descriptor_t<TeamSize,
                                         DatasetBlockDim,
                                         PqBits,
                                         PqLen,
                                         CodeBookT,
                                         DataT,
                                         IndexT,
                                         DistanceT>(encoded_dataset_ptr,
                                                    encoded_dataset_dim,
                                                    n_subspace,
                                                    vq_code_book_ptr,
                                                    vq_scale,
                                                    pq_code_book_ptr,
                                                    pq_scale,
                                                    size,
                                                    dim);
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodeBookT,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename DatasetIdxT>
auto vpq_dataset_descriptor_init(const vpq_dataset<CodeBookT, DatasetIdxT>& dataset,
                                 rmm::cuda_stream_view stream)
  -> dataset_descriptor_host<DataT, IndexT, DistanceT>
{
  const float vq_scale = 1.0f;
  const float pq_scale = 1.0f;
  cagra_q_dataset_descriptor_t<TeamSize,
                               DatasetBlockDim,
                               PqBits,
                               PqLen,
                               CodeBookT,
                               DataT,
                               IndexT,
                               DistanceT>
    dd_host{dataset.data.data_handle(),
            dataset.encoded_row_length(),
            dataset.pq_dim(),
            dataset.vq_code_book.data_handle(),
            vq_scale,
            dataset.pq_code_book.data_handle(),
            pq_scale,
            IndexT(dataset.n_rows()),
            dataset.dim()};
  dataset_descriptor_host<DataT, IndexT, DistanceT> result{
    dd_host, stream, TeamSize, DatasetBlockDim};
  vpq_dataset_descriptor_init_kernel<TeamSize,
                                     DatasetBlockDim,
                                     PqBits,
                                     PqLen,
                                     CodeBookT,
                                     DataT,
                                     IndexT,
                                     DistanceT><<<1, 1, 0, stream>>>(result.dev_ptr,
                                                                     dd_host.encoded_dataset_ptr,
                                                                     dd_host.encoded_dataset_dim,
                                                                     dd_host.n_subspace,
                                                                     dd_host.vq_code_book_ptr,
                                                                     dd_host.vq_scale,
                                                                     dd_host.pq_code_book_ptr,
                                                                     dd_host.pq_scale,
                                                                     dd_host.size,
                                                                     dd_host.dim);
  return result;
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename DatasetIdxT>
auto vpq_dataset_descriptor_init_runtime(const vpq_dataset<half, DatasetIdxT>& dataset,
                                         rmm::cuda_stream_view stream)

{
  if (dataset.pq_bits() == 8) {
    if (dataset.pq_len() == 2) {
      return vpq_dataset_descriptor_init<TeamSize,
                                         DatasetBlockDim,
                                         8,
                                         2,
                                         half,
                                         DataT,
                                         IndexT,
                                         DistanceT>(dataset, stream);
    } else if (dataset.pq_len() == 4) {
      return vpq_dataset_descriptor_init<TeamSize,
                                         DatasetBlockDim,
                                         8,
                                         4,
                                         half,
                                         DataT,
                                         IndexT,
                                         DistanceT>(dataset, stream);
    } else {
      RAFT_FAIL("Subspace dimension must be 2 or 4");
    }
  } else {
    RAFT_FAIL("Only 8-bit PQ is supported now");
  }
}

template <typename DataT, typename IndexT, typename DistanceT, typename DatasetIdxT>
auto dataset_descriptor_init(const vpq_dataset<half, DatasetIdxT>& dataset,
                             rmm::cuda_stream_view stream)
  -> dataset_descriptor_host<DataT, IndexT, DistanceT>
{
  constexpr int64_t max_dataset_block_dim = 512;
  int64_t dataset_block_dim               = 128;
  while (dataset_block_dim < dataset.dim() && dataset_block_dim < max_dataset_block_dim) {
    dataset_block_dim *= 2;
  }
  switch (dataset_block_dim) {
    case 128:
      return vpq_dataset_descriptor_init_runtime<8, 128, DataT, IndexT, DistanceT, DatasetIdxT>(
        dataset, stream);
    case 256:
      return vpq_dataset_descriptor_init_runtime<16, 256, DataT, IndexT, DistanceT, DatasetIdxT>(
        dataset, stream);
    default:
      return vpq_dataset_descriptor_init_runtime<32, 512, DataT, IndexT, DistanceT, DatasetIdxT>(
        dataset, stream);
  }
}

}  // namespace cuvs::neighbors::cagra::detail
