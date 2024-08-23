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

#include <raft/util/vectorized.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
struct alignas(device::LOAD_128BIT_T) standard_dataset_descriptor_t
  : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using LOAD_T    = device::LOAD_128BIT_T;
  using QUERY_T   = float;
  using base_type::dim;
  using base_type::smem_ws_size_in_bytes;
  using typename base_type::compute_distance_type;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::setup_workspace_type;
  constexpr static inline auto kTeamSize        = TeamSize;
  constexpr static inline auto kDatasetBlockDim = DatasetBlockDim;

  const DATA_T* ptr;
  uint32_t ld;

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
                TeamSize,
                get_smem_ws_size_in_bytes(dim)),
      ptr(ptr),
      ld(ld)
  {
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
  const typename DescriptorT::base_type* that,
  void* smem_ptr,
  const typename DescriptorT::DATA_T* queries_ptr,
  uint32_t query_id) -> const typename DescriptorT::base_type*
{
  using descriptor_type          = DescriptorT;
  using base_type                = typename DescriptorT::base_type;
  using QUERY_T                  = typename descriptor_type::QUERY_T;
  constexpr auto DatasetBlockDim = DescriptorT::kDatasetBlockDim;
  using word_type                = uint32_t;
  if (((void*)that) != smem_ptr) {
    constexpr auto kStructWords = sizeof(DescriptorT) / sizeof(word_type);
    auto* dst                   = reinterpret_cast<word_type*>(smem_ptr);
    auto* src                   = reinterpret_cast<const word_type*>(that);
    for (unsigned i = threadIdx.x; i < kStructWords; i += blockDim.x) {
      dst[i] = src[i];
    }
  }

  uint32_t dim = that->dim;
  auto buf = reinterpret_cast<QUERY_T*>(reinterpret_cast<uint8_t*>(smem_ptr) + sizeof(DescriptorT));
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

  return const_cast<const base_type*>(reinterpret_cast<base_type*>(smem_ptr));
}

template <typename DescriptorT>
_RAFT_DEVICE __noinline__ auto compute_distance_standard(
  const typename DescriptorT::base_type* desc_,
  typename DescriptorT::INDEX_T dataset_index,
  cuvs::distance::DistanceType metric,
  bool valid) -> typename DescriptorT::DISTANCE_T
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using DISTANCE_T                = typename DescriptorT::DISTANCE_T;
  using INDEX_T                   = typename DescriptorT::INDEX_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;

  auto* __restrict__ desc      = reinterpret_cast<const DescriptorT*>(desc_);
  auto* __restrict__ query_ptr = reinterpret_cast<const QUERY_T*>(desc + 1);
  const auto dataset_ptr       = desc->ptr + (static_cast<std::uint64_t>(desc->ld) * dataset_index);
  const unsigned lane_id       = threadIdx.x % kTeamSize;
  auto dim                     = desc->dim;

  DISTANCE_T norm2 = 0;
  if (valid) {
    for (uint32_t elem_offset = 0; elem_offset < dim; elem_offset += kDatasetBlockDim) {
      constexpr unsigned vlen      = device::get_vlen<LOAD_T, DATA_T>();
      constexpr unsigned reg_nelem = raft::ceildiv<unsigned>(kDatasetBlockDim, kTeamSize * vlen);
      raft::TxN_t<DATA_T, vlen> dl_buff[reg_nelem];
#pragma unroll
      for (uint32_t e = 0; e < reg_nelem; e++) {
        const uint32_t k = (lane_id + (kTeamSize * e)) * vlen + elem_offset;
        if (k >= dim) break;
        dl_buff[e].load(dataset_ptr, k);
      }
#pragma unroll
      for (uint32_t e = 0; e < reg_nelem; e++) {
        const uint32_t k = (lane_id + (kTeamSize * e)) * vlen + elem_offset;
        if (k >= dim) break;
#pragma unroll
        for (uint32_t v = 0; v < vlen; v++) {
          // Note this loop can go above the dataset_dim for padded arrays. This is not a problem
          // because:
          // - Above the last element (dataset_dim-1), the query array is filled with zeros.
          // - The data buffer has to be also padded with zeros.
          DISTANCE_T d;
          raft::lds(d, query_ptr + device::swizzling(k + v));
          constexpr cuvs::spatial::knn::detail::utils::mapping<float> mapping{};
          switch (metric) {
            case cuvs::distance::DistanceType::L2Expanded:
              d -= mapping(dl_buff[e].val.data[v]);
              norm2 += d * d;
              break;
            case cuvs::distance::DistanceType::InnerProduct:
              norm2 -= d * mapping(dl_buff[e].val.data[v]);
              break;
          }
        }
      }
    }
  }
#pragma unroll
  for (uint32_t offset = kTeamSize / 2; offset > 0; offset >>= 1) {
    norm2 += __shfl_xor_sync(0xffffffff, norm2, offset);
  }
  return norm2;
}

template <uint32_t TeamSize,
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
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  new (out) desc_type(&setup_workspace_standard<desc_type>,
                      &compute_distance_standard<desc_type>,
                      ptr,
                      size,
                      dim,
                      ld);
}

template <uint32_t TeamSize,
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
    standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;
  static const void* init_kernel;

  template <typename DatasetT>
  static auto init(const cagra::search_params& params,
                   const DatasetT& dataset,
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
      {&result.dev_ptr, &dd_host.ptr, &dd_host.size, &dd_host.dim, &dd_host.ld};
    RAFT_CUDA_TRY(cudaLaunchKernel(init_kernel, 1, 1, args, 0, stream));
    return result;
  }

  template <typename DatasetT>
  static auto priority(const cagra::search_params& params, const DatasetT& dataset) -> double
  {
    // If explicit team_size is specified and doesn't match the instance, discard it
    if (params.team_size != 0 && TeamSize != params.team_size) { return -1.0; }
    // Otherwise, favor the closest dataset dimensionality.
    return 1.0 / (0.1 + std::abs(double(dataset.dim()) - double(DatasetBlockDim)));
  }
};

}  // namespace cuvs::neighbors::cagra::detail
