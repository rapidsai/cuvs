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
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/integer_utils.hpp>

namespace cuvs::neighbors::cagra::detail {

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PQ_BITS,
          uint32_t PQ_LEN,
          typename CodeBookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct cagra_q_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type   = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using CODE_BOOK_T = CodeBookT;
  using QUERY_T     = half;
  using base_type::args;
  using base_type::extra_ptr3;
  using typename base_type::args_t;
  using typename base_type::compute_distance_type;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::LOAD_T;
  using typename base_type::setup_workspace_type;
  constexpr static inline auto kMetric          = Metric;
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;
  constexpr static inline auto kPqBits          = PQ_BITS;
  constexpr static inline auto kPqLen           = PQ_LEN;

  static_assert(std::is_same_v<CODE_BOOK_T, half>, "Only CODE_BOOK_T = `half` is supported now");

  // alignas(LOAD_T) const std::uint8_t* encoded_dataset_ptr;
  // const CODE_BOOK_T* vq_code_book_ptr;
  // const CODE_BOOK_T* pq_code_book_ptr;
  // std::uint32_t encoded_dataset_dim;
  // std::uint32_t n_subspace;

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
  RAFT_INLINE_FUNCTION static constexpr auto n_subspace(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word2;
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
  RAFT_INLINE_FUNCTION static constexpr auto n_subspace(const args_t& args) noexcept
    -> const uint32_t&
  {
    return args.extra_word2;
  }

  static constexpr std::uint32_t kSMemCodeBookSizeInBytes =
    (1 << PQ_BITS) * PQ_LEN * utils::size_of<CODE_BOOK_T>();

  _RAFT_HOST_DEVICE cagra_q_dataset_descriptor_t(setup_workspace_type* setup_workspace_impl,
                                                 compute_distance_type* compute_distance_impl,
                                                 const std::uint8_t* encoded_dataset_ptr,
                                                 std::uint32_t encoded_dataset_dim,
                                                 std::uint32_t n_subspace,
                                                 const CODE_BOOK_T* vq_code_book_ptr,
                                                 const CODE_BOOK_T* pq_code_book_ptr,
                                                 std::size_t size,
                                                 std::uint32_t dim)
    : base_type(setup_workspace_impl,
                compute_distance_impl,
                size,
                dim,
                TeamSize,
                get_smem_ws_size_in_bytes(dim))
  {
    cagra_q_dataset_descriptor_t::encoded_dataset_ptr(args) = encoded_dataset_ptr;
    cagra_q_dataset_descriptor_t::vq_code_book_ptr(args)    = vq_code_book_ptr;
    this->pq_code_book_ptr()                                = pq_code_book_ptr;
    cagra_q_dataset_descriptor_t::encoded_dataset_dim(args) = encoded_dataset_dim;
    cagra_q_dataset_descriptor_t::n_subspace(args)          = n_subspace;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(cagra_q_dataset_descriptor_t) == alignof(base_type));
  }

 private:
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
};

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_vpq(const DescriptorT* that,
                                                   void* smem_ptr,
                                                   const typename DescriptorT::DATA_T* queries_ptr,
                                                   uint32_t query_id) -> const DescriptorT*
{
  using base_type                = typename DescriptorT::base_type;
  using DATA_T                   = typename DescriptorT::DATA_T;
  using DISTANCE_T               = typename DescriptorT::DISTANCE_T;
  using INDEX_T                  = typename DescriptorT::INDEX_T;
  using LOAD_T                   = typename DescriptorT::LOAD_T;
  using QUERY_T                  = typename DescriptorT::QUERY_T;
  using CODE_BOOK_T              = typename DescriptorT::CODE_BOOK_T;
  constexpr auto TeamSize        = DescriptorT::kTeamSize;
  constexpr auto DatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto PQ_BITS         = DescriptorT::kPqBits;
  constexpr auto PQ_LEN          = DescriptorT::kPqLen;

  auto* r = reinterpret_cast<DescriptorT*>(smem_ptr);

  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(LOAD_T);
    using blob_type           = LOAD_T[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }

    auto codebook_buf = uint32_t(__cvta_generic_to_shared(r + 1));
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(LOAD_T))) {
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

  constexpr cuvs::spatial::knn::detail::utils::mapping<half> mapping{};
  auto smem_query_ptr =
    reinterpret_cast<const QUERY_T*>(reinterpret_cast<uint8_t*>(smem_ptr) + sizeof(DescriptorT) +
                                     DescriptorT::kSMemCodeBookSizeInBytes);
  for (unsigned i = threadIdx.x * 2; i < dim; i += blockDim.x * 2) {
    half2 buf2{0, 0};
    if (i < dim) { buf2.x = mapping(queries_ptr[i]); }
    if (i + 1 < dim) { buf2.y = mapping(queries_ptr[i + 1]); }
    if constexpr ((PQ_BITS == 8) && (PQ_LEN % 2 == 0)) {
      // Use swizzling in the condition to reduce bank conflicts in shared
      // memory, which are likely to occur when pq_code_book_dim is large.
      ((half2*)smem_query_ptr)[device::swizzling<std::uint32_t, DatasetBlockDim / 2>(i / 2)] = buf2;
    } else {
      (reinterpret_cast<half2*>(smem_query_ptr + i))[0] = buf2;
    }
  }

  return const_cast<const DescriptorT*>(r);
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_vpq(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  using DISTANCE_T               = typename DescriptorT::DISTANCE_T;
  using LOAD_T                   = typename DescriptorT::LOAD_T;
  using QUERY_T                  = typename DescriptorT::QUERY_T;
  using CODE_BOOK_T              = typename DescriptorT::CODE_BOOK_T;
  constexpr auto TeamSize        = DescriptorT::kTeamSize;
  constexpr auto DatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto PQ_BITS         = DescriptorT::kPqBits;
  constexpr auto PQ_LEN          = DescriptorT::kPqLen;

  const uint32_t pq_codebook_ptr = args.smem_ws_ptr;
  const uint32_t query_ptr       = pq_codebook_ptr + DescriptorT::kSMemCodeBookSizeInBytes;
  const auto* __restrict__ node_ptr =
    DescriptorT::encoded_dataset_ptr(args) +
    (static_cast<std::uint64_t>(DescriptorT::encoded_dataset_dim(args)) * dataset_index);
  const unsigned lane_id = threadIdx.x % TeamSize;
  // const uint32_t& vq_code                  = *reinterpret_cast<const std::uint32_t*>(node_ptr);
  uint32_t vq_code;
  raft::ldg(vq_code, reinterpret_cast<const std::uint32_t*>(node_ptr));
  static_assert(PQ_BITS == 8, "Only pq_bits == 8 is supported at the moment.");
  DISTANCE_T norm = 0;
  for (uint32_t elem_offset = 0; elem_offset < args.dim; elem_offset += DatasetBlockDim) {
    constexpr unsigned vlen = 4;  // **** DO NOT CHANGE ****
    constexpr unsigned nelem =
      raft::div_rounding_up_unsafe<unsigned>(DatasetBlockDim / PQ_LEN, TeamSize * vlen);
    // Loading PQ codes
    uint32_t pq_codes[nelem];
#pragma unroll
    for (std::uint32_t e = 0; e < nelem; e++) {
      const std::uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset / PQ_LEN;
      if (k >= DescriptorT::n_subspace(args)) break;
      // Loading 4 x 8-bit PQ-codes using 32-bit load ops (from device memory)
      raft::ldg(pq_codes[e], reinterpret_cast<const std::uint32_t*>(node_ptr + 4 + k));
    }
    //
    if constexpr (PQ_LEN % 2 == 0) {
      // **** Use half2 for distance computation ****
#pragma unroll 1
      for (std::uint32_t e = 0; e < nelem; e++) {
        const std::uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset / PQ_LEN;
        if (k >= DescriptorT::n_subspace(args)) break;
        // Loading VQ code-book
        raft::TxN_t<half2, vlen / 2> vq_vals[PQ_LEN];
#pragma unroll
        for (std::uint32_t m = 0; m < PQ_LEN; m += 1) {
          const uint32_t d = (vlen * m) + (PQ_LEN * k);
          if (d >= args.dim) break;
          vq_vals[m].load(reinterpret_cast<const half2*>(DescriptorT::vq_code_book_ptr(args) + d +
                                                         (args.dim * vq_code)),
                          0);
        }
        // Compute distance
        std::uint32_t pq_code = pq_codes[e];
#pragma unroll
        for (std::uint32_t v = 0; v < vlen; v++) {
          if (PQ_LEN * (v + k) >= args.dim) break;
#pragma unroll
          for (std::uint32_t m = 0; m < PQ_LEN; m += 2) {
            const std::uint32_t d1 = m + (PQ_LEN * v);
            const std::uint32_t d  = d1 + (PQ_LEN * k);
            half2 q2, c2;
            // Loading query vector from smem
            device::lds(q2,
                        query_ptr + sizeof(uint32_t) *
                                      device::swizzling<std::uint32_t, DatasetBlockDim / 2>(d / 2));
            // Loading PQ code book from smem
            device::lds(c2,
                        pq_codebook_ptr + sizeof(CODE_BOOK_T) * ((1 << PQ_BITS) * 2 * (m / 2) +
                                                                 (2 * (pq_code & 0xff))));
            // L2 distance
            auto dist = q2 - c2 - vq_vals[d1 / vlen].val.data[(d1 % vlen) / 2];
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
        const std::uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset / PQ_LEN;
        if (k >= DescriptorT::n_subspace(args)) break;
        // Loading VQ code-book
        raft::TxN_t<CODE_BOOK_T, vlen> vq_vals[PQ_LEN];
#pragma unroll
        for (std::uint32_t m = 0; m < PQ_LEN; m++) {
          const std::uint32_t d = (vlen * m) + (PQ_LEN * k);
          if (d >= args.dim) break;
          // Loading 4 x 8/16-bit VQ-values using 32/64-bit load ops (from L2$ or device
          // memory)
          vq_vals[m].load(reinterpret_cast<const half2*>(DescriptorT::vq_code_book_ptr(args) + d +
                                                         (args.dim * vq_code)),
                          0);
        }
        // Compute distance
        std::uint32_t pq_code = pq_codes[e];
#pragma unroll
        for (std::uint32_t v = 0; v < vlen; v++) {
          if (PQ_LEN * (v + k) >= args.dim) break;
          raft::TxN_t<CODE_BOOK_T, PQ_LEN> pq_vals;
          device::lds(*pq_vals.vectorized_data(),
                      pq_codebook_ptr + sizeof(CODE_BOOK_T) * PQ_LEN * (pq_code & 0xff));
#pragma unroll
          for (std::uint32_t m = 0; m < PQ_LEN; m++) {
            const std::uint32_t d1 = m + (PQ_LEN * v);
            const std::uint32_t d  = d1 + (PQ_LEN * k);
            // if (d >= dataset_dim) break;
            DISTANCE_T diff;
            device::lds(diff, query_ptr + sizeof(QUERY_T) * d);
            diff -= static_cast<DISTANCE_T>(pq_vals.data[m]);
            diff -= static_cast<DISTANCE_T>(vq_vals[d1 / vlen].val.data[d1 % vlen]);
            norm += diff * diff;
          }
          pq_code >>= 8;
        }
      }
    }
  }
  return norm;
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
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
                                          const CodeBookT* pq_code_book_ptr,
                                          std::size_t size,
                                          std::uint32_t dim)
{
  using desc_type = cagra_q_dataset_descriptor_t<Metric,
                                                 TeamSize,
                                                 DatasetBlockDim,
                                                 PqBits,
                                                 PqLen,
                                                 CodeBookT,
                                                 DataT,
                                                 IndexT,
                                                 DistanceT>;
  using base_type = typename desc_type::base_type;
  new (out) desc_type(
    reinterpret_cast<typename base_type::setup_workspace_type*>(&setup_workspace_vpq<desc_type>),
    reinterpret_cast<typename base_type::compute_distance_type*>(&compute_distance_vpq<desc_type>),
    encoded_dataset_ptr,
    encoded_dataset_dim,
    n_subspace,
    vq_code_book_ptr,
    pq_code_book_ptr,
    size,
    dim);
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          uint32_t PqBits,
          uint32_t PqLen,
          typename CodebookT,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct vpq_descriptor_spec : public instance_spec<DataT, IndexT, DistanceT> {
  using base_type = instance_spec<DataT, IndexT, DistanceT>;
  using typename base_type::data_type;
  using typename base_type::distance_type;
  using typename base_type::host_type;
  using typename base_type::index_type;

  template <typename DatasetT>
  constexpr static inline auto accepts_dataset()
    -> std::enable_if_t<is_vpq_dataset_v<DatasetT>, bool>
  {
    return std::is_same_v<typename DatasetT::math_type, CodebookT>;
  }

  template <typename DatasetT>
  constexpr static inline auto accepts_dataset()
    -> std::enable_if_t<!is_vpq_dataset_v<DatasetT>, bool>
  {
    return false;
  }

  using descriptor_type = cagra_q_dataset_descriptor_t<Metric,
                                                       TeamSize,
                                                       DatasetBlockDim,
                                                       PqBits,
                                                       PqLen,
                                                       CodebookT,
                                                       DataT,
                                                       IndexT,
                                                       DistanceT>;
  static const void* init_kernel;

  template <typename DatasetT>
  static auto init(const cagra::search_params& params,
                   const DatasetT& dataset,
                   cuvs::distance::DistanceType metric,
                   rmm::cuda_stream_view stream) -> host_type
  {
    descriptor_type dd_host{nullptr,
                            nullptr,
                            dataset.data.data_handle(),
                            dataset.encoded_row_length(),
                            dataset.pq_dim(),
                            dataset.vq_code_book.data_handle(),
                            dataset.pq_code_book.data_handle(),
                            IndexT(dataset.n_rows()),
                            dataset.dim()};
    host_type result{dd_host, stream, DatasetBlockDim};
    void* args[] =  // NOLINT
      {&result.dev_ptr,
       &descriptor_type::encoded_dataset_ptr(dd_host.args),
       &descriptor_type::encoded_dataset_dim(dd_host.args),
       &descriptor_type::n_subspace(dd_host.args),
       &descriptor_type::vq_code_book_ptr(dd_host.args),
       &dd_host.pq_code_book_ptr(),
       &dd_host.size,
       &dd_host.args.dim};
    RAFT_CUDA_TRY(cudaLaunchKernel(init_kernel, 1, 1, args, 0, stream));
    return result;
  }

  template <typename DatasetT>
  static auto priority(const cagra::search_params& params,
                       const DatasetT& dataset,
                       cuvs::distance::DistanceType metric) -> double
  {
    // If explicit team_size is specified and doesn't match the instance, discard it
    if (params.team_size != 0 && TeamSize != params.team_size) { return -1.0; }
    if (cuvs::distance::DistanceType::L2Expanded != metric) { return -1.0; }
    // Match codebook params
    if (dataset.pq_bits() != PqBits) { return -1.0; }
    if (dataset.pq_len() != PqLen) { return -1.0; }
    // Otherwise, favor the closest dataset dimensionality.
    return 1.0 / (0.1 + std::abs(double(dataset.dim()) - double(DatasetBlockDim)));
  }
};

}  // namespace cuvs::neighbors::cagra::detail
