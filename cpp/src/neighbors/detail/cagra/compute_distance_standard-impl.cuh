/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "compute_distance_standard.hpp"

#include <cuvs/distance/distance.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/pow2_utils.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {
namespace {
template <typename T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(T a, T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::L2Expanded, T>
{
  T diff = a - b;
  return diff * diff;
}

template <typename T, cuvs::distance::DistanceType Metric>
RAFT_DEVICE_INLINE_FUNCTION constexpr auto dist_op(T a, T b)
  -> std::enable_if_t<Metric == cuvs::distance::DistanceType::InnerProduct, T>
{
  return -a * b;
}
}  // namespace

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct standard_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using QUERY_T   = float;
  using base_type::args;
  using base_type::smem_ws_size_in_bytes;
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

  static constexpr RAFT_INLINE_FUNCTION auto ptr(const args_t& args) noexcept
    -> const DATA_T* const&
  {
    return (const DATA_T* const&)(args.extra_ptr1);
  }
  static constexpr RAFT_INLINE_FUNCTION auto ptr(args_t& args) noexcept -> const DATA_T*&
  {
    return (const DATA_T*&)(args.extra_ptr1);
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
                                                  uint32_t ld)
    : base_type(setup_workspace_impl,
                compute_distance_impl,
                size,
                dim,
                raft::Pow2<TeamSize>::Log2,
                get_smem_ws_size_in_bytes(dim))
  {
    standard_dataset_descriptor_t::ptr(args) = ptr;
    standard_dataset_descriptor_t::ld(args)  = ld;
    static_assert(sizeof(*this) == sizeof(base_type));
    static_assert(alignof(standard_dataset_descriptor_t) == alignof(base_type));
  }

 private:
  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    return sizeof(standard_dataset_descriptor_t) +
           raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }
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
      buf[j] = 0.0;
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
        // Note this loop can go above the dataset_dim for padded arrays. This is not a problem
        // because:
        // - Above the last element (dataset_dim-1), the query array is filled with zeros.
        // - The data buffer has to be also padded with zeros.
        DISTANCE_T d;
        device::lds(
          d,
          query_smem_ptr +
            sizeof(QUERY_T) * device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(k + v));
        r += dist_op<DISTANCE_T, DescriptorT::kMetric>(
          d, cuvs::spatial::knn::detail::utils::mapping<DISTANCE_T>{}(data[e][v]));
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
  return compute_distance_standard_worker<DescriptorT>(
    DescriptorT::ptr(args) + (static_cast<std::uint64_t>(DescriptorT::ld(args)) * dataset_index),
    args.dim,
    args.smem_ws_ptr);
}

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
                                          uint32_t ld)
{
  using desc_type =
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  using base_type = typename desc_type::base_type;
  new (out) desc_type(reinterpret_cast<typename base_type::setup_workspace_type*>(
                        &setup_workspace_standard<desc_type>),
                      reinterpret_cast<typename base_type::compute_distance_type*>(
                        &compute_distance_standard<desc_type>),
                      ptr,
                      size,
                      dim,
                      ld);
}

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
  rmm::cuda_stream_view stream)
{
  using desc_type =
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  using base_type = typename desc_type::base_type;
  desc_type dd_host{nullptr, nullptr, ptr, size, dim, ld};
  host_type result{dd_host, stream};

  standard_dataset_descriptor_init_kernel<Metric,
                                          TeamSize,
                                          DatasetBlockDim,
                                          DataT,
                                          IndexT,
                                          DistanceT>
    <<<1, 1, 0, stream>>>(result.dev_ptr(), ptr, size, dim, desc_type::ld(dd_host.args));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  return result;
}

}  // namespace cuvs::neighbors::cagra::detail
