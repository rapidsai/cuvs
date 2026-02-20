/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_distance_vpq.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT
#if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)
          ,
          cuvs::distance::DistanceType Metric
#else
          ,
          typename QueryT
#endif
          >
struct cagra_q_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type   = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using CODE_BOOK_T = CodebookT;
#if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)
  using QUERY_T = half;
#else
  using QUERY_T = QueryT;
#endif
  using base_type::args;
  using base_type::extra_ptr3;
  using typename base_type::args_t;
  using typename base_type::compute_distance_type;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::LOAD_T;
  using typename base_type::setup_workspace_type;
#if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)
  constexpr static inline auto kMetric = Metric;
#endif
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;
  constexpr static inline auto kPqBits          = PQ_BITS;
  constexpr static inline auto kPqLen           = PQ_LEN;

  static_assert(std::is_same_v<CODE_BOOK_T, half>, "Only CODE_BOOK_T = `half` is supported now");

  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_ptr(args_t& args) noexcept
    -> const uint8_t*&
  {
    return (const uint8_t*&)args.extra_ptr1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto vq_code_book_ptr(args_t& args) noexcept
    -> const CODE_BOOK_T*&
  {
    return (const CODE_BOOK_T*&)args.extra_ptr2;
  }
  RAFT_INLINE_FUNCTION constexpr auto pq_code_book_ptr() noexcept -> const CODE_BOOK_T*&
  {
    return (const CODE_BOOK_T*&)extra_ptr3;
  }
  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_dim(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word1;
  }

  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_ptr(const args_t& args) noexcept
    -> const uint8_t* const&
  {
    return (const uint8_t*&)args.extra_ptr1;
  }
  RAFT_INLINE_FUNCTION static constexpr auto vq_code_book_ptr(const args_t& args) noexcept
    -> const CODE_BOOK_T* const&
  {
    return (const CODE_BOOK_T*&)args.extra_ptr2;
  }
  RAFT_INLINE_FUNCTION constexpr auto pq_code_book_ptr() const noexcept -> const CODE_BOOK_T* const&
  {
    return (const CODE_BOOK_T*&)extra_ptr3;
  }
  RAFT_INLINE_FUNCTION static constexpr auto encoded_dataset_dim(const args_t& args) noexcept
    -> const uint32_t&
  {
    return args.extra_word1;
  }

  static constexpr std::uint32_t kSMemCodeBookSizeInBytes =
    (1 << PQ_BITS) * PQ_LEN * utils::size_of<CODE_BOOK_T>();

  _RAFT_HOST_DEVICE cagra_q_dataset_descriptor_t(setup_workspace_type* setup_workspace_impl,
                                                 compute_distance_type* compute_distance_impl,
                                                 const std::uint8_t* encoded_dataset_ptr,
                                                 std::uint32_t encoded_dataset_dim,
                                                 const CODE_BOOK_T* vq_code_book_ptr,
                                                 const CODE_BOOK_T* pq_code_book_ptr,
                                                 IndexT size,
                                                 std::uint32_t dim)
    : base_type(setup_workspace_impl,
                compute_distance_impl,
                size,
                dim,
                raft::Pow2<TeamSize>::Log2,
                get_smem_ws_size_in_bytes(dim))
  {
    cagra_q_dataset_descriptor_t::encoded_dataset_ptr(args) = encoded_dataset_ptr;
    cagra_q_dataset_descriptor_t::vq_code_book_ptr(args)    = vq_code_book_ptr;
    this->pq_code_book_ptr()                                = pq_code_book_ptr;
    cagra_q_dataset_descriptor_t::encoded_dataset_dim(args) = encoded_dataset_dim;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(cagra_q_dataset_descriptor_t) == alignof(base_type));
  }

  // Public static method to compute smem size without constructing descriptor
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    /* SMEM workspace layout:
      1. The descriptor itself
      2. Codebook (kSMemCodeBookSizeInBytes bytes)
      3. Queries (smem_query_buffer_length elems)
    */
    return sizeof(cagra_q_dataset_descriptor_t) + kSMemCodeBookSizeInBytes +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }

 private:
};

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

  return const_cast<const DescriptorT*>(r);
}

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
            // if (d >= dataset_dim) break;
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

#ifndef BUILD_KERNEL
// The init kernel is not needed when building JIT fragments (BUILD_KERNEL is defined)
// It's only needed for non-JIT initialization. When BUILD_KERNEL is defined, we're building
// a JIT fragment and don't want this kernel to be instantiated.
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
RAFT_KERNEL __launch_bounds__(1, 1)
  vpq_dataset_descriptor_init_kernel(dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
                                     const std::uint8_t* encoded_dataset_ptr,
                                     uint32_t encoded_dataset_dim,
                                     const CodebookT* vq_code_book_ptr,
                                     const CodebookT* pq_code_book_ptr,
                                     IndexT size,
                                     uint32_t dim)
{
  using desc_type = cagra_q_dataset_descriptor_t<TeamSize,
                                                 DatasetBlockDim,
                                                 PqBits,
                                                 PqLen,
                                                 CodebookT,
                                                 DataT,
                                                 IndexT,
                                                 DistanceT
#if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)
                                                 ,
                                                 Metric
#else
                                                 ,
                                                 half
#endif
                                                 >;
  using base_type = typename desc_type::base_type;
#ifdef CUVS_ENABLE_JIT_LTO
  // For JIT, we don't use the function pointers, so set them to nullptr
  // The free functions are called directly instead
  new (out) desc_type(nullptr,  // setup_workspace_impl - not used in JIT
                      nullptr,  // compute_distance_impl - not used in JIT
                      encoded_dataset_ptr,
                      encoded_dataset_dim,
                      vq_code_book_ptr,
                      pq_code_book_ptr,
                      size,
                      dim);
#else
  // For CUDA 12 (non-JIT), set the function pointers properly
  new (out) desc_type(
    reinterpret_cast<typename base_type::setup_workspace_type*>(&setup_workspace_vpq<desc_type>),
    reinterpret_cast<typename base_type::compute_distance_type*>(&compute_distance_vpq<desc_type>),
    encoded_dataset_ptr,
    encoded_dataset_dim,
    vq_code_book_ptr,
    pq_code_book_ptr,
    size,
    dim);
#endif
}
#endif  // #ifndef BUILD_KERNEL

#ifndef BUILD_KERNEL
// The init_ function is not needed when building JIT fragments (BUILD_KERNEL is defined)
// It's only needed for non-JIT initialization. When BUILD_KERNEL is defined, we're building
// a JIT fragment and don't want this host function to be included.
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
dataset_descriptor_host<DataT, IndexT, DistanceT>
vpq_descriptor_spec<Metric,
                    TeamSize,
                    DatasetBlockDim,
                    PqBits,
                    PqLen,
                    CodebookT,
                    DataT,
                    IndexT,
                    DistanceT>::init_(const cagra::search_params& params,
                                      const std::uint8_t* encoded_dataset_ptr,
                                      uint32_t encoded_dataset_dim,
                                      const CodebookT* vq_code_book_ptr,
                                      const CodebookT* pq_code_book_ptr,
                                      IndexT size,
                                      uint32_t dim)
{
  using desc_type = cagra_q_dataset_descriptor_t<TeamSize,
                                                 DatasetBlockDim,
                                                 PqBits,
                                                 PqLen,
                                                 CodebookT,
                                                 DataT,
                                                 IndexT,
                                                 DistanceT
#if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)
                                                 ,
                                                 Metric
#else
                                                 ,
                                                 half
#endif
                                                 >;
  using base_type = typename desc_type::base_type;

  return host_type{desc_type{nullptr,
                             nullptr,
                             encoded_dataset_ptr,
                             encoded_dataset_dim,
                             vq_code_book_ptr,
                             pq_code_book_ptr,
                             size,
                             dim},
                   [=](dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr,
                       rmm::cuda_stream_view stream) {
                     // Use init kernel for both JIT and CUDA 12
                     // The kernel handles JIT vs non-JIT via ifdef internally
                     vpq_dataset_descriptor_init_kernel<Metric,
                                                        TeamSize,
                                                        DatasetBlockDim,
                                                        PqBits,
                                                        PqLen,
                                                        CodebookT,
                                                        DataT,
                                                        IndexT,
                                                        DistanceT>
                       <<<1, 1, 0, stream>>>(dev_ptr,
                                             encoded_dataset_ptr,
                                             encoded_dataset_dim,
                                             vq_code_book_ptr,
                                             pq_code_book_ptr,
                                             size,
                                             dim);
                     RAFT_CUDA_TRY(cudaPeekAtLastError());
                   },
                   Metric,
                   DatasetBlockDim,
                   true,    // is_vpq
                   PqBits,  // pq_bits
                   PqLen};  // pq_len
}
#endif  // #ifndef BUILD_KERNEL

}  // namespace cuvs::neighbors::cagra::detail
