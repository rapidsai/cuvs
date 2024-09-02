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

#include "compute_distance.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/logger-macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

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

  // const DATA_T* ptr;
  // uint32_t ld;

  // RAFT_INLINE_FUNCTION constexpr auto ptr() noexcept -> const DATA_T*&
  // {
  //   return (const DATA_T*&)(extra_ptr1);
  // }

  // RAFT_INLINE_FUNCTION constexpr auto ptr() const noexcept -> const DATA_T* const&
  // {
  //   return (const DATA_T* const&)(extra_ptr1);
  // }

  // RAFT_INLINE_FUNCTION constexpr auto ld() noexcept -> uint32_t& { return extra_word1; }
  // RAFT_INLINE_FUNCTION constexpr auto ld() const noexcept -> const uint32_t& { return
  // extra_word1; }

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
  using base_type                = typename DescriptorT::base_type;
  using QUERY_T                  = typename DescriptorT::QUERY_T;
  using LOAD_T                   = typename DescriptorT::LOAD_T;
  constexpr auto DatasetBlockDim = DescriptorT::kDatasetBlockDim;
  auto* r                        = reinterpret_cast<DescriptorT*>(smem_ptr);
  auto* buf                      = reinterpret_cast<QUERY_T*>(r + 1);
  if (r != that) {
    constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(LOAD_T);
    using blob_type           = LOAD_T[kCount];
    auto& src                 = reinterpret_cast<const blob_type&>(*that);
    auto& dst                 = reinterpret_cast<blob_type&>(*r);
    for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
      dst[i] = src[i];
    }
    const auto smem_ptr_offset =
      reinterpret_cast<uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<uint8_t*>(r);
    if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(LOAD_T))) {
      r->args.smem_ws_ptr = uint32_t(__cvta_generic_to_shared(buf));
    }
    __syncthreads();
  }

  uint32_t dim = r->args.dim;
  auto buf_len = raft::round_up_safe<uint32_t>(dim, DatasetBlockDim);
  queries_ptr += dim * query_id;
  for (unsigned i = threadIdx.x; i < buf_len; i += blockDim.x) {
    unsigned j = device::swizzling(i);
    if (i < dim) {
      buf[j] = cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(queries_ptr[i]);
    } else {
      buf[j] = 0.0;
    }
  }

  return const_cast<const DescriptorT*>(r);
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_standard(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using DISTANCE_T                = typename DescriptorT::DISTANCE_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;

  // const auto* __restrict__ query_ptr = reinterpret_cast<const QUERY_T*>(args.smem_ws_ptr);
  const auto* __restrict__ dataset_ptr =
    DescriptorT::ptr(args) + (static_cast<std::uint64_t>(DescriptorT::ld(args)) * dataset_index);
  const auto lane_id = threadIdx.x % kTeamSize;

  DISTANCE_T r = 0;
  for (uint32_t elem_offset = 0; elem_offset < args.dim; elem_offset += kDatasetBlockDim) {
    constexpr unsigned vlen      = device::get_vlen<LOAD_T, DATA_T>();
    constexpr unsigned reg_nelem = raft::ceildiv<unsigned>(kDatasetBlockDim, kTeamSize * vlen);
    raft::TxN_t<DATA_T, vlen> dl_buff[reg_nelem];
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = (lane_id + (kTeamSize * e)) * vlen + elem_offset;
      if (k >= args.dim) break;
      dl_buff[e].load(dataset_ptr, k);
    }
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = (lane_id + (kTeamSize * e)) * vlen + elem_offset;
      if (k >= args.dim) break;
#pragma unroll
      for (uint32_t v = 0; v < vlen; v++) {
        // Note this loop can go above the dataset_dim for padded arrays. This is not a problem
        // because:
        // - Above the last element (dataset_dim-1), the query array is filled with zeros.
        // - The data buffer has to be also padded with zeros.
        DISTANCE_T d;
        device::lds(d, args.smem_ws_ptr + sizeof(QUERY_T) * device::swizzling(k + v));
        r += dist_op<DISTANCE_T, DescriptorT::kMetric>(
          d, cuvs::spatial::knn::detail::utils::mapping<DISTANCE_T>{}(dl_buff[e].val.data[v]));
      }
    }
  }
  return r;
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__launch_bounds__(1, 1) __global__ void standard_dataset_descriptor_init_kernel(
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* out,
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
struct standard_descriptor_spec : public instance_spec<DataT, IndexT, DistanceT> {
  using base_type = instance_spec<DataT, IndexT, DistanceT>;
  using typename base_type::data_type;
  using typename base_type::distance_type;
  using typename base_type::host_type;
  using typename base_type::index_type;

  template <typename DatasetT>
  constexpr static inline bool accepts_dataset()
  {
    return is_strided_dataset_v<DatasetT>;
  }

  using descriptor_type =
    standard_dataset_descriptor_t<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  static const void* init_kernel;

  template <typename DatasetT>
  static auto init(const cagra::search_params& params,
                   const DatasetT& dataset,
                   cuvs::distance::DistanceType metric,
                   rmm::cuda_stream_view stream) -> host_type
  {
    descriptor_type dd_host{nullptr,
                            nullptr,
                            dataset.view().data_handle(),
                            IndexT(dataset.n_rows()),
                            dataset.dim(),
                            dataset.stride()};
    host_type result{dd_host, stream, DatasetBlockDim};
    void* args[] =  // NOLINT
      {&result.dev_ptr,
       &descriptor_type::ptr(dd_host.args),
       &dd_host.size,
       &dd_host.args.dim,
       &descriptor_type::ld(dd_host.args)};
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
    if (Metric != metric) { return -1.0; }
    // Otherwise, favor the closest dataset dimensionality.
    return 1.0 / (0.1 + std::abs(double(dataset.dim()) - double(DatasetBlockDim)));
  }
};

}  // namespace cuvs::neighbors::cagra::detail
