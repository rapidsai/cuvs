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

// TODO: This shouldn't be invoking spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/vectorized.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT = float>
struct standard_dataset_descriptor_t : public dataset_descriptor_base_t<DataT, IndexT, DistanceT> {
  using base_type = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  using LOAD_T    = device::LOAD_128BIT_T;
  using QUERY_T   = float;
  using base_type::dim;
  using base_type::smem_ws_size_in_bytes;
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::ws_handle;

  const DATA_T* ptr;
  size_t ld;

  _RAFT_HOST_DEVICE standard_dataset_descriptor_t(const DATA_T* ptr,
                                                  INDEX_T size,
                                                  uint32_t dim,
                                                  size_t ld)
    : base_type(size, dim, TeamSize, get_smem_ws_size_in_bytes(dim)), ptr(ptr), ld(ld)
  {
    base_type::template assert_struct_size<sizeof(*this)>();
  }

  _RAFT_DEVICE [[nodiscard]] auto set_smem_ws(void* smem_ptr) const -> ws_handle
  {
    return reinterpret_cast<ws_handle>(smem_ptr);
  }

  _RAFT_DEVICE void copy_query(ws_handle smem_workspace, const DATA_T* query_ptr) const
  {
    auto buf     = smem_query_buffer(smem_workspace);
    auto buf_len = smem_ws_size_in_bytes / sizeof(QUERY_T);
    for (unsigned i = threadIdx.x; i < buf_len; i += blockDim.x) {
      unsigned j = device::swizzling(i);
      if (i < dim) {
        buf[j] = cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(query_ptr[i]);
      } else {
        buf[j] = 0.0;
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

 private:
  template <typename T, cuvs::distance::DistanceType METRIC>
  RAFT_DEVICE_INLINE_FUNCTION constexpr static auto dist_op(T a, T b)
    -> std::enable_if_t<METRIC == cuvs::distance::DistanceType::L2Expanded, T>
  {
    T diff = a - b;
    return diff * diff;
  }

  template <typename T, cuvs::distance::DistanceType METRIC>
  RAFT_DEVICE_INLINE_FUNCTION constexpr static auto dist_op(T a, T b)
    -> std::enable_if_t<METRIC == cuvs::distance::DistanceType::InnerProduct, T>
  {
    return -a * b;
  }

  template <cuvs::distance::DistanceType METRIC>
  RAFT_DEVICE_INLINE_FUNCTION auto compute_similarity(ws_handle smem_workspace,
                                                      const INDEX_T dataset_i,
                                                      const bool valid) const -> DISTANCE_T
  {
    auto query_ptr         = smem_query_buffer(smem_workspace);
    const auto dataset_ptr = ptr + dataset_i * ld;
    const unsigned lane_id = threadIdx.x % TeamSize;

    DISTANCE_T norm2 = 0;
    if (valid) {
      for (uint32_t elem_offset = 0; elem_offset < dim; elem_offset += DatasetBlockDim) {
        constexpr unsigned vlen      = device::get_vlen<LOAD_T, DATA_T>();
        constexpr unsigned reg_nelem = raft::ceildiv<unsigned>(DatasetBlockDim, TeamSize * vlen);
        raft::TxN_t<DATA_T, vlen> dl_buff[reg_nelem];
#pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
          const uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset;
          if (k >= dim) break;
          dl_buff[e].load(dataset_ptr, k);
        }
#pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
          const uint32_t k = (lane_id + (TeamSize * e)) * vlen + elem_offset;
          if (k >= dim) break;
#pragma unroll
          for (uint32_t v = 0; v < vlen; v++) {
            const uint32_t kv = k + v;
            // Note this loop can go above the dataset_dim for padded arrays. This is not a problem
            // because:
            // - Above the last element (dataset_dim-1), the query array is filled with zeros.
            // - The data buffer has to be also padded with zeros.
            DISTANCE_T d = query_ptr[device::swizzling(kv)];
            norm2 += dist_op<DISTANCE_T, METRIC>(
              d, cuvs::spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].val.data[v]));
          }
        }
      }
    }
#pragma unroll
    for (uint32_t offset = TeamSize / 2; offset > 0; offset >>= 1) {
      norm2 += __shfl_xor_sync(0xffffffff, norm2, offset);
    }
    return norm2;
  }

  RAFT_DEVICE_INLINE_FUNCTION constexpr auto smem_query_buffer(ws_handle smem_workspace) const
    -> QUERY_T*
  {
    return reinterpret_cast<QUERY_T*>(smem_workspace);
  }

  RAFT_INLINE_FUNCTION constexpr static auto get_smem_ws_size_in_bytes(uint32_t dim) -> uint32_t
  {
    return raft::round_up_safe<uint32_t>(dim, DatasetBlockDim) * sizeof(QUERY_T);
  }
};

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
  size_t ld)
{
  new (out) standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>(
    ptr, size, dim, ld);
}

template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename DatasetT>
auto standard_dataset_descriptor_init(const DatasetT& dataset, rmm::cuda_stream_view stream)
  -> dataset_descriptor_host<DataT, IndexT, DistanceT>
{
  standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT> dd_host{
    dataset.view().data_handle(), IndexT(dataset.n_rows()), dataset.dim(), dataset.stride()};
  dataset_descriptor_host<DataT, IndexT, DistanceT> result{dd_host, stream, DatasetBlockDim};
  standard_dataset_descriptor_init_kernel<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>
    <<<1, 1, 0, stream>>>(result.dev_ptr, dd_host.ptr, dd_host.size, dd_host.dim, dd_host.ld);
  return result;
}

template <typename DatasetT, typename ReturnT>
using enable_strided = std::enable_if_t<is_strided_dataset_v<DatasetT>, ReturnT>;

template <typename DataT, typename IndexT, typename DistanceT, typename DatasetT>
auto dataset_descriptor_init(const DatasetT& dataset, rmm::cuda_stream_view stream)
  -> enable_strided<DatasetT, dataset_descriptor_host<DataT, IndexT, DistanceT>>
{
  constexpr int64_t max_dataset_block_dim = 256;
  int64_t dataset_block_dim               = 64;
  while (dataset_block_dim < dataset.dim() && dataset_block_dim < max_dataset_block_dim) {
    dataset_block_dim *= 2;
  }
  switch (dataset_block_dim) {
    case 64:
      return standard_dataset_descriptor_init<8, 64, DataT, IndexT, DistanceT>(dataset, stream);
    case 128:
      return standard_dataset_descriptor_init<16, 128, DataT, IndexT, DistanceT>(dataset, stream);
    default:
      return standard_dataset_descriptor_init<32, 256, DataT, IndexT, DistanceT>(dataset, stream);
  }
}

// template <typename DataT, typename IndexT, typename DistanceT>
// struct descriptor_instance_spec {
//   template <uint32_t TeamSize, uint32_t DatasetBlockDim>
//   struct standard_descriptor {
//     template <typename DatasetT>
//     struct dataset_instance_spec {
//       static auto init(const cagra::search_params& params,
//                        const DatasetT& dataset,
//                        rmm::cuda_stream_view stream)
//         -> dataset_descriptor_host<DataT, IndexT, DistanceT>
//       {
//         standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>
//         dd_host{
//           dataset.view().data_handle(), IndexT(dataset.n_rows()), dataset.dim(),
//           dataset.stride()};
//         dataset_descriptor_host<DataT, IndexT, DistanceT> result{dd_host, stream,
//         DatasetBlockDim}; standard_dataset_descriptor_init_kernel<TeamSize, DatasetBlockDim,
//         DataT, IndexT, DistanceT>
//           <<<1, 1, 0, stream>>>(result.dev_ptr, dd_host.ptr, dd_host.size, dd_host.dim,
//           dd_host.ld);
//         return result;
//       }

//       static auto error(const cagra::search_params& params,
//                         const DatasetT& dataset,
//                         rmm::cuda_stream_view stream)
//         -> dataset_descriptor_host<DataT, IndexT, DistanceT>
//       {
//         RAFT_FAIL("Invalid team_size {%u} - no kernel instance found for this value.");
//       }

//       static auto priority(const cagra::search_params& params, const DatasetT& dataset) -> double
//       {
//         // If explicit team_size is specified and doesn't match the instance, discard it
//         if (params.team_size != 0 && TeamSize != params.team_size) { return -1.0; }
//         // Otherwise, favor the closest dataset dimensionality.
//         return 1.0 / (0.1 + std::abs(dataset.dim() - DatasetBlockDim));
//       }
//     };
//   };
// };

// template <typename... Specs>
// struct instance_selector {
//   template <typename ReturnT, typename DatasetT>
//   static auto init(const cagra::search_params& params,
//                    const DatasetT& dataset,
//                    rmm::cuda_stream_view stream) -> ReturnT = 0;
// };

// template <typename Spec>
// struct instance_selector<Spec> {
//   template <typename DatasetT>
//   static auto select_worker(const cagra::search_params& params, const DatasetT& dataset)
//   {
//     auto p = Spec::priority(params, dataset);
//     return std::make_tuple(p >= 0 ? &(Spec::init) : &(Spec::error), p);
//   }
// }

// template <typename Spec0, typename... Specs>
// struct instance_selector<Spec, Specs...> {
//   template <typename DatasetT>
//   static auto select_worker(const cagra::search_params& params, const DatasetT& dataset)
//   {
//     auto p0  = Spec::priority(params, dataset);
//     auto sel = instance_selector<Specs...>::select_worker<DatasetT>(params, dataset);
//     return p0 > std::get<double>(sel) ? std::make_tuple(&(Spec::init), p0) : sel;
//   }
// }

// template <typename DatasetT, typename... Specs>

// template <typename DataT, typename IndexT, typename DistanceT, typename DatasetIdxT>
// auto dataset_descriptor_init(const cagra::search_params& params,
//                              const strided_dataset<DataT, DatasetIdxT>& dataset,
//                              rmm::cuda_stream_view stream)
//   -> dataset_descriptor_host<DataT, IndexT, DistanceT>
// {
// }

}  // namespace cuvs::neighbors::cagra::detail
