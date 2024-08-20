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

#include "device_common.hpp"
#include "hashmap.hpp"
#include "utils.hpp"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <raft/core/logger-macros.hpp>
#include <raft/core/operators.hpp>

// TODO: This shouldn't be invoking spatial/knn
#include "../ann_utils.cuh"

#include <raft/util/vectorized.cuh>

#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

template <typename DataT, typename IndexT, typename DistanceT>
struct dataset_descriptor_base_t {
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  /**
   * Maximum expected size of the descriptor struct.
   * This covers all standard and VPQ descriptors; we need this to copy the descriptor from global
   * memory. Increase this if new fields are needed (but try to keep the descriptors small really).
   */
  static constexpr size_t kMaxStructSize = 72;

  template <size_t ActualSize, size_t MaximumSize = kMaxStructSize>
  static inline constexpr void assert_struct_size()
  {
    static_assert(ActualSize <= MaximumSize,
                  "The maximum descriptor size is tracked in the dataset_descriptor_base_t. "
                  "Update this constant if implementing a new, larger descriptor.");
  }

  struct distance_workspace;
  using ws_handle = distance_workspace*;

  /** Number of records in the database. */
  INDEX_T size;
  /** Dimensionality of the data/queries. */
  uint32_t dim;
  /** How many threads are involved in computing a single distance. */
  uint32_t team_size;
  /** Total dynamic shared memory required by the descriptor.  */
  uint32_t smem_ws_size_in_bytes;

  _RAFT_HOST_DEVICE dataset_descriptor_base_t(INDEX_T size,
                                              uint32_t dim,
                                              uint32_t team_size,
                                              uint32_t smem_ws_size_in_bytes)
    : size(size), dim(dim), team_size(team_size), smem_ws_size_in_bytes(smem_ws_size_in_bytes)
  {
  }

  RAFT_DEVICE_INLINE_FUNCTION void copy_descriptor_per_block(
    dataset_descriptor_base_t* target) const
  {
    using word_type             = uint32_t;
    constexpr auto kStructWords = kMaxStructSize / sizeof(word_type);
    auto* dst                   = reinterpret_cast<word_type*>(target);
    auto* src                   = reinterpret_cast<const word_type*>(this);
    for (unsigned i = threadIdx.x; i < kStructWords; i += blockDim.x) {
      dst[i] = src[i];
    }
    __syncthreads();
  }

  /** Setup the shared memory workspace (e.g. assign pointers or prepare a lookup table). */
  _RAFT_DEVICE [[nodiscard]] virtual auto set_smem_ws(void* smem_ptr) const -> ws_handle = 0;

  /** Copy the query to the shared memory. */
  _RAFT_DEVICE virtual void copy_query(ws_handle smem_workspace, const DATA_T* query_ptr) const = 0;

  /** Compute the distance from the query vector (stored in the smem_workspace) and a dataset vector
   * given by the dataset_index. */
  _RAFT_DEVICE virtual auto compute_distance(ws_handle smem_workspace,
                                             INDEX_T dataset_index,
                                             cuvs::distance::DistanceType metric,
                                             bool valid) const -> DISTANCE_T = 0;
};

template <typename DataT, typename IndexT, typename DistanceT>
struct dataset_descriptor_host {
  using dev_descriptor_t         = dataset_descriptor_base_t<DataT, IndexT, DistanceT>;
  dev_descriptor_t* dev_ptr      = nullptr;
  uint32_t smem_ws_size_in_bytes = 0;
  uint32_t team_size             = 0;
  uint32_t dataset_block_dim     = 0;

  template <typename DescriptorImpl>
  dataset_descriptor_host(const DescriptorImpl& dd_host,
                          rmm::cuda_stream_view stream,
                          uint32_t dataset_block_dim)
    : stream_{stream},
      smem_ws_size_in_bytes{dd_host.smem_ws_size_in_bytes},
      team_size{dd_host.team_size},
      dataset_block_dim{dataset_block_dim}
  {
    RAFT_CUDA_TRY(cudaMallocAsync(&dev_ptr, dev_descriptor_t::kMaxStructSize, stream_));
  }

  ~dataset_descriptor_host() noexcept
  {
    if (dev_ptr == nullptr) { return; }
    RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(dev_ptr, stream_));
  }

  dataset_descriptor_host(dataset_descriptor_host&& other)
  {
    std::swap(this->dev_ptr, other.dev_ptr);
    std::swap(this->smem_ws_size_in_bytes, other.smem_ws_size_in_bytes);
    std::swap(this->stream_, other.stream_);
    std::swap(this->team_size, other.team_size);
    std::swap(this->dataset_block_dim, other.dataset_block_dim);
  }
  dataset_descriptor_host& operator=(dataset_descriptor_host&& b)
  {
    auto& a = *this;
    std::swap(a.dev_ptr, b.dev_ptr);
    std::swap(a.smem_ws_size_in_bytes, b.smem_ws_size_in_bytes);
    std::swap(a.stream_, b.stream_);
    std::swap(a.team_size, b.team_size);
    std::swap(a.dataset_block_dim, b.dataset_block_dim);
    return a;
  }
  dataset_descriptor_host(const dataset_descriptor_host&)            = delete;
  dataset_descriptor_host& operator=(const dataset_descriptor_host&) = delete;

 private:
  rmm::cuda_stream_view stream_;
};

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

extern template struct standard_dataset_descriptor_t<8, 128, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, float, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, half, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, int8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, uint8_t, uint32_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, float, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<8, 128, half, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<16, 256, half, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 512, half, uint64_t, float>;
extern template struct standard_dataset_descriptor_t<32, 1024, half, uint64_t, float>;

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
          typename DatasetIdxT>
auto standard_dataset_descriptor_init(const strided_dataset<DataT, DatasetIdxT>& dataset,
                                      rmm::cuda_stream_view stream)
  -> dataset_descriptor_host<DataT, IndexT, DistanceT>
{
  standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT> dd_host{
    dataset.view().data_handle(), IndexT(dataset.n_rows()), dataset.dim(), dataset.stride()};
  dataset_descriptor_host<DataT, IndexT, DistanceT> result{dd_host, stream, DatasetBlockDim};
  standard_dataset_descriptor_init_kernel<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>
    <<<1, 1, 0, stream>>>(result.dev_ptr, dd_host.ptr, dd_host.size, dd_host.dim, dd_host.ld);
  return result;
}

template <typename DataT, typename IndexT, typename DistanceT, typename DatasetIdxT>
auto dataset_descriptor_init(const strided_dataset<DataT, DatasetIdxT>& dataset,
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
      return standard_dataset_descriptor_init<8, 128, DataT, IndexT, DistanceT, DatasetIdxT>(
        dataset, stream);
    case 256:
      return standard_dataset_descriptor_init<16, 256, DataT, IndexT, DistanceT, DatasetIdxT>(
        dataset, stream);
    default:
      return standard_dataset_descriptor_init<32, 512, DataT, IndexT, DistanceT, DatasetIdxT>(
        dataset, stream);
  }
}

}  // namespace cuvs::neighbors::cagra::detail
