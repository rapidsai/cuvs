/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_distance_standard.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

#if defined(CUVS_ENABLE_JIT_LTO) || defined(BUILD_KERNEL)

// When JIT LTO is enabled or building kernel fragments, dist_op is an extern function that gets JIT
// linked from fragments Each fragment provides a metric-specific implementation (L2Expanded,
// InnerProduct, etc.) The planner will link the appropriate fragment based on the metric Note:
// extern functions cannot be constexpr, so we remove constexpr here Note: These are in the detail
// namespace (not anonymous) so they can be found by JIT linking
// QueryT can be float (for most metrics) or uint8_t (for BitwiseHamming)
template <typename QUERY_T, typename DISTANCE_T>
extern __device__ DISTANCE_T dist_op(QUERY_T a, QUERY_T b);

// Normalization is also JIT linked from fragments (no-op for most metrics, cosine normalization for
// CosineExpanded) The planner will link the appropriate fragment (cosine or noop) based on the
// metric
// QueryT is needed to match the descriptor template signature (always float for normalization)
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

#endif

namespace {

#if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)

// When JIT LTO is disabled, dist_op is a template function with Metric as a template parameter
template <typename DATA_T, typename DISTANCE_T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(DATA_T a, DATA_T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::L2Expanded, DISTANCE_T>
{
  DISTANCE_T diff = a - b;
  return diff * diff;
}

template <typename DATA_T, typename DISTANCE_T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(DATA_T a, DATA_T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::InnerProduct ||
                        Metric == cuvs::distance::DistanceType::CosineExpanded,
                      DISTANCE_T>
{
  return -static_cast<DISTANCE_T>(a) * static_cast<DISTANCE_T>(b);
}

template <typename DATA_T, typename DISTANCE_T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(DATA_T a, DATA_T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::BitwiseHamming &&
                        std::is_integral_v<DATA_T>,
                      DISTANCE_T>
{
  // mask the result of xor for the integer promotion
  const auto v = (a ^ b) & 0xffu;
  return __popc(v);
}

#endif  // #if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)
}  // namespace

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
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
struct standard_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
#if !defined(CUVS_ENABLE_JIT_LTO) && !defined(BUILD_KERNEL)
  // When JIT LTO is disabled, Metric is a template parameter
  using QUERY_T = typename std::
    conditional_t<Metric == cuvs::distance::DistanceType::BitwiseHamming, DataT, float>;
#else
  // When JIT LTO is enabled, QueryT is passed as a template parameter
  using QUERY_T = QueryT;
#endif
  using base_type::args;
  using base_type::smem_ws_size_in_bytes;
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

  static constexpr RAFT_INLINE_FUNCTION auto ptr(const args_t& args) noexcept
    -> const DATA_T* const&
  {
    return (const DATA_T* const&)(args.extra_ptr1);
  }
  static constexpr RAFT_INLINE_FUNCTION auto ptr(args_t& args) noexcept -> const DATA_T*&
  {
    return (const DATA_T*&)(args.extra_ptr1);
  }

  static constexpr RAFT_INLINE_FUNCTION auto dataset_norms_ptr(const args_t& args) noexcept
    -> const DISTANCE_T* const&
  {
    return (const DISTANCE_T* const&)(args.extra_ptr2);
  }
  static constexpr RAFT_INLINE_FUNCTION auto dataset_norms_ptr(args_t& args) noexcept
    -> const DISTANCE_T*&
  {
    return (const DISTANCE_T*&)(args.extra_ptr2);
  }

  static constexpr RAFT_INLINE_FUNCTION auto ld(const args_t& args) noexcept -> const uint32_t&
  {
    return args.extra_word1;
  }
  static constexpr RAFT_INLINE_FUNCTION auto ld(args_t& args) noexcept -> uint32_t&
  {
    return args.extra_word1;
  }

  _RAFT_HOST_DEVICE standard_dataset_descriptor_t(setup_workspace_type* setup_workspace_impl,
                                                  compute_distance_type* compute_distance_impl,
                                                  const DATA_T* ptr,
                                                  INDEX_T size,
                                                  uint32_t dim,
                                                  uint32_t ld,
                                                  const DISTANCE_T* dataset_norms = nullptr)
    : base_type(setup_workspace_impl,
                compute_distance_impl,
                size,
                dim,
                raft::Pow2<TeamSize>::Log2,
                get_smem_ws_size_in_bytes(dim))
  {
    standard_dataset_descriptor_t::ptr(args)               = ptr;
    standard_dataset_descriptor_t::ld(args)                = ld;
    standard_dataset_descriptor_t::dataset_norms_ptr(args) = dataset_norms;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(standard_dataset_descriptor_t) == alignof(base_type));
  }

  // Public static method to compute smem size without constructing descriptor
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    return sizeof(standard_dataset_descriptor_t) +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }

 private:
};

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto setup_workspace_standard(
  const DescriptorT* that,
  void* smem_ptr,
  const typename DescriptorT::DATA_T* queries_ptr,
  uint32_t query_id) -> const DescriptorT*
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using base_type                 = typename DescriptorT::base_type;
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
  return const_cast<const DescriptorT*>(r);
}

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
        // because:
        // - Above the last element (dataset_dim-1), the query array is filled with zeros.
        // - The data buffer has to be also padded with zeros.
        QUERY_T d;
        device::lds(
          d,
          query_smem_ptr +
            sizeof(QUERY_T) * device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(k + v));
#if defined(CUVS_ENABLE_JIT_LTO) || defined(BUILD_KERNEL)
        // When JIT LTO is enabled or building kernel fragments, dist_op is an extern function (no
        // template parameters)
        r += dist_op<QUERY_T, DISTANCE_T>(
          d, cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(data[e][v]));
#else
        // When JIT LTO is disabled, dist_op is a template function with Metric parameter
        r += dist_op<QUERY_T, DISTANCE_T, DescriptorT::kMetric>(
          d, cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(data[e][v]));
#endif
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

#if defined(CUVS_ENABLE_JIT_LTO) || defined(BUILD_KERNEL)
  // Normalization is JIT linked from fragments (no-op or cosine normalization)
  // The planner links the appropriate fragment based on the metric
  distance =
    apply_normalization_standard<DescriptorT::kTeamSize,
                                 DescriptorT::kDatasetBlockDim,
                                 typename DescriptorT::DATA_T,
                                 typename DescriptorT::INDEX_T,
                                 typename DescriptorT::DISTANCE_T,
                                 typename DescriptorT::QUERY_T>(distance, args, dataset_index);
#else
  // When JIT LTO is disabled, kMetric is always available as a compile-time constant
  if constexpr (DescriptorT::kMetric == cuvs::distance::DistanceType::CosineExpanded) {
    const auto* dataset_norms = DescriptorT::dataset_norms_ptr(args);
    auto norm                 = dataset_norms[dataset_index];
    if (norm > 0) { distance = distance / norm; }
  }
#endif

  return distance;
}

#ifndef BUILD_KERNEL
// The init kernel is used for both JIT and non-JIT initialization
// When BUILD_KERNEL is defined, we're building a JIT fragment and don't want this kernel.
// The kernel handles JIT vs non-JIT via ifdef internally
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
RAFT_KERNEL __launch_bounds__(1, 1)
  standard_dataset_descriptor_init_kernel(dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
                                          const DataT* ptr,
                                          IndexT size,
                                          uint32_t dim,
                                          uint32_t ld,
                                          const DistanceT* dataset_norms = nullptr)
{
#if !defined(CUVS_ENABLE_JIT_LTO)
  // When JIT LTO is disabled, Metric is a template parameter (last parameter)
  using desc_type =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, Metric>;
  using base_type = typename desc_type::base_type;

  // For CUDA 12 (non-JIT), set the function pointers properly
  new (out) desc_type(reinterpret_cast<typename base_type::setup_workspace_type*>(
                        &setup_workspace_standard<desc_type>),
                      reinterpret_cast<typename base_type::compute_distance_type*>(
                        &compute_distance_standard<desc_type>),
                      ptr,
                      size,
                      dim,
                      ld,
                      dataset_norms);
#else
  // When JIT LTO is enabled, Metric is not a template parameter
  using query_t =
    std::conditional_t<Metric == cuvs::distance::DistanceType::BitwiseHamming, DataT, float>;
  using desc_type =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, query_t>;
  using base_type = typename desc_type::base_type;

  // For JIT, we don't use the function pointers, so set them to nullptr
  // The free functions are called directly instead
  new (out) desc_type(nullptr,  // setup_workspace_impl - not used in JIT
                      nullptr,  // compute_distance_impl - not used in JIT
                      ptr,
                      size,
                      dim,
                      ld,
                      dataset_norms);
#endif
}
#endif  // #ifndef BUILD_KERNEL

#ifndef BUILD_KERNEL
// The init_ function is used for both JIT and non-JIT initialization
// When BUILD_KERNEL is defined, we're building a JIT fragment and don't want this function.
template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
dataset_descriptor_host<DataT, IndexT, DistanceT>
standard_descriptor_spec<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>::init_(
  const cagra::search_params& params,
  const DataT* ptr,
  IndexT size,
  uint32_t dim,
  uint32_t ld,
  const DistanceT* dataset_norms)
{
#if !defined(CUVS_ENABLE_JIT_LTO)
  // When JIT LTO is disabled, Metric is a template parameter (last parameter)
  using desc_type =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, Metric>;
  using base_type = typename desc_type::base_type;
#else
  // When JIT LTO is enabled, Metric is not a template parameter
  // QueryT depends on metric: uint8_t for BitwiseHamming, float for others
  using query_t =
    std::conditional_t<Metric == cuvs::distance::DistanceType::BitwiseHamming, DataT, float>;
  using desc_type =
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, query_t>;
  using base_type = typename desc_type::base_type;
#endif

  RAFT_EXPECTS(Metric != cuvs::distance::DistanceType::CosineExpanded || dataset_norms != nullptr,
               "Dataset norms must be provided for CosineExpanded metric");

  return host_type{desc_type{nullptr, nullptr, ptr, size, dim, ld, dataset_norms},
                   [=](dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr,
                       rmm::cuda_stream_view stream) {
                     // Use init kernel for both JIT and CUDA 12
                     // The kernel handles JIT vs non-JIT via ifdef internally
                     standard_dataset_descriptor_init_kernel<Metric,
                                                             TeamSize,
                                                             DatasetBlockDim,
                                                             DataT,
                                                             IndexT,
                                                             DistanceT>
                       <<<1, 1, 0, stream>>>(dev_ptr, ptr, size, dim, ld, dataset_norms);
                     RAFT_CUDA_TRY(cudaPeekAtLastError());
                   },
                   Metric,
                   DatasetBlockDim,
                   false,  // is_vpq
                   0,      // pq_bits
                   0};     // pq_len
}
#endif  // #ifndef BUILD_KERNEL

}  // namespace cuvs::neighbors::cagra::detail
