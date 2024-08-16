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
namespace device {

// using LOAD_256BIT_T = ulonglong4;
using LOAD_128BIT_T = uint4;
using LOAD_64BIT_T  = uint64_t;

template <class LOAD_T, class DATA_T>
RAFT_DEVICE_INLINE_FUNCTION constexpr unsigned get_vlen()
{
  return utils::size_of<LOAD_T>() / utils::size_of<DATA_T>();
}

template <unsigned TeamSize,
          unsigned DatasetBlockDim,
          class DATASET_DESCRIPTOR_T,
          class DISTANCE_T,
          class INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_nodes(
  INDEX_T* const result_indices_ptr,       // [num_pickup]
  DISTANCE_T* const result_distances_ptr,  // [num_pickup]
  typename DATASET_DESCRIPTOR_T::ws_handle workspace,
  const DATASET_DESCRIPTOR_T& dataset_desc,
  const size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const INDEX_T* const seed_ptr,  // [num_seeds]
  const uint32_t num_seeds,
  INDEX_T* const visited_hash_ptr,
  const uint32_t hash_bitlen,
  const cuvs::distance::DistanceType metric,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1)
{
  uint32_t max_i = num_pickup;
  if (max_i % (32 / TeamSize)) { max_i += (32 / TeamSize) - (max_i % (32 / TeamSize)); }

  for (uint32_t i = threadIdx.x / TeamSize; i < max_i; i += blockDim.x / TeamSize) {
    const bool valid_i = (i < num_pickup);

    INDEX_T best_index_team_local;
    DISTANCE_T best_norm2_team_local = utils::get_max_value<DISTANCE_T>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      // Select a node randomly and compute the distance to it
      INDEX_T seed_index;
      if (valid_i) {
        // uint32_t gid = i + (num_pickup * (j + (num_distilation * block_id)));
        uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % dataset_desc.size;
        }
      }

      DISTANCE_T norm2;
      switch (metric) {
        case cuvs::distance::DistanceType::L2Expanded:
          norm2 =
            dataset_desc.template compute_similarity<cuvs::distance::DistanceType::L2Expanded>(
              workspace, seed_index, valid_i);
          break;
        case cuvs::distance::DistanceType::InnerProduct:
          norm2 =
            dataset_desc.template compute_similarity<cuvs::distance::DistanceType::InnerProduct>(
              workspace, seed_index, valid_i);
          break;
        default: break;
      }

      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x % TeamSize;
    if (valid_i && lane_id == 0) {
      if (hashmap::insert(visited_hash_ptr, hash_bitlen, best_index_team_local)) {
        result_distances_ptr[i] = best_norm2_team_local;
        result_indices_ptr[i]   = best_index_team_local;
      } else {
        result_distances_ptr[i] = utils::get_max_value<DISTANCE_T>();
        result_indices_ptr[i]   = utils::get_max_value<INDEX_T>();
      }
    }
  }
}

template <unsigned TeamSize,
          unsigned DatasetBlockDim,
          typename DATASET_DESCRIPTOR_T,
          typename DISTANCE_T,
          typename INDEX_T>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_child_nodes(
  INDEX_T* result_child_indices_ptr,
  DISTANCE_T* result_child_distances_ptr,
  // query
  typename DATASET_DESCRIPTOR_T::ws_handle workspace,
  // [dataset_dim, dataset_size]
  const DATASET_DESCRIPTOR_T& dataset_desc,
  // [knn_k, dataset_size]
  const INDEX_T* knn_graph,
  uint32_t knn_k,
  // hashmap
  INDEX_T* visited_hashmap_ptr,
  uint32_t hash_bitlen,
  const INDEX_T* parent_indices,
  const INDEX_T* internal_topk_list,
  uint32_t search_width,
  cuvs::distance::DistanceType metric)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

  // Read child indices of parents from knn graph and check if the distance
  // computaiton is necessary.
  for (uint32_t i = threadIdx.x; i < knn_k * search_width; i += blockDim.x) {
    const INDEX_T smem_parent_id = parent_indices[i / knn_k];
    INDEX_T child_id             = invalid_index;
    if (smem_parent_id != invalid_index) {
      const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
      child_id             = knn_graph[(i % knn_k) + (static_cast<int64_t>(knn_k) * parent_id)];
    }
    if (child_id != invalid_index) {
      if (hashmap::insert(visited_hashmap_ptr, hash_bitlen, child_id) == 0) {
        child_id = invalid_index;
      }
    }
    result_child_indices_ptr[i] = child_id;
  }
  __syncthreads();

  // Compute the distance to child nodes
  uint32_t max_i = knn_k * search_width;
  if (max_i % (32 / TeamSize)) { max_i += (32 / TeamSize) - (max_i % (32 / TeamSize)); }
  for (uint32_t tid = threadIdx.x; tid < max_i * TeamSize; tid += blockDim.x) {
    const auto i       = tid / TeamSize;
    const bool valid_i = (i < (knn_k * search_width));
    INDEX_T child_id   = invalid_index;
    if (valid_i) { child_id = result_child_indices_ptr[i]; }

    DISTANCE_T norm2;
    switch (metric) {
      case cuvs::distance::DistanceType::L2Expanded:
        norm2 = dataset_desc.template compute_similarity<cuvs::distance::DistanceType::L2Expanded>(
          workspace, child_id, child_id != invalid_index);
        break;
      case cuvs::distance::DistanceType::InnerProduct:
        norm2 =
          dataset_desc.template compute_similarity<cuvs::distance::DistanceType::InnerProduct>(
            workspace, child_id, child_id != invalid_index);
        break;
      default: break;
    }

    // Store the distance
    const unsigned lane_id = threadIdx.x % TeamSize;
    if (valid_i && lane_id == 0) {
      if (child_id != invalid_index) {
        result_child_distances_ptr[i] = norm2;
      } else {
        result_child_distances_ptr[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
  }
}

}  // namespace device

template <typename DataT, typename IndexT, typename DistanceT>
struct dataset_descriptor_base_t {
  using DATA_T     = DataT;
  using INDEX_T    = IndexT;
  using DISTANCE_T = DistanceT;

  struct distance_workspace;
  using ws_handle = distance_workspace*;

  INDEX_T size;
  uint32_t dim;

  _RAFT_HOST_DEVICE dataset_descriptor_base_t(INDEX_T size, uint32_t dim) : size(size), dim(dim) {}

  /** Total dynamic shared memory required by the descriptor.  */
  _RAFT_HOST_DEVICE [[nodiscard]] virtual auto smem_ws_size_in_bytes() const -> uint32_t = 0;

  /** Set shared memory workspace (pointers). */
  _RAFT_DEVICE [[nodiscard]] virtual auto set_smem_ws(void* smem_ptr) const -> ws_handle = 0;

  /** Copy the query to the shared memory. */
  _RAFT_DEVICE virtual void copy_query(ws_handle smem_workspace, const DATA_T* query_ptr) const = 0;

  _RAFT_DEVICE virtual void compute_distance_to_random_nodes(
    ws_handle smem_workspace,
    INDEX_T* const result_indices_ptr,       // [num_pickup]
    DISTANCE_T* const result_distances_ptr,  // [num_pickup]
    const size_t num_pickup,
    const unsigned num_distilation,
    const uint64_t rand_xor_mask,
    const INDEX_T* const seed_ptr,  // [num_seeds]
    const uint32_t num_seeds,
    INDEX_T* const visited_hash_ptr,
    const uint32_t hash_bitlen,
    const cuvs::distance::DistanceType metric,
    const uint32_t block_id   = 0,
    const uint32_t num_blocks = 1) const = 0;

  _RAFT_DEVICE virtual void compute_distance_to_child_nodes(
    ws_handle smem_workspace,
    INDEX_T* const result_child_indices_ptr,
    DISTANCE_T* const result_child_distances_ptr,
    // [knn_k, dataset_size]
    const INDEX_T* const knn_graph,
    const uint32_t knn_k,
    // hashmap
    INDEX_T* const visited_hashmap_ptr,
    const uint32_t hash_bitlen,
    const INDEX_T* const parent_indices,
    const INDEX_T* const internal_topk_list,
    const uint32_t search_width,
    const cuvs::distance::DistanceType metric) const = 0;
};

template <typename DataT, typename IndexT, typename DistanceT>
struct dataset_descriptor_host {
  dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dev_ptr = nullptr;
  uint32_t smem_ws_size_in_bytes                               = 0;
  uint32_t team_size                                           = 0;
  uint32_t dataset_block_dim                                   = 0;

  template <typename DescriptorImpl>
  dataset_descriptor_host(const DescriptorImpl& dd_host,
                          rmm::cuda_stream_view stream,
                          uint32_t team_size,
                          uint32_t dataset_block_dim)
    : stream_{stream},
      smem_ws_size_in_bytes{dd_host.smem_ws_size_in_bytes()},
      team_size{team_size},
      dataset_block_dim{dataset_block_dim}
  {
    RAFT_CUDA_TRY(cudaMallocAsync(&dev_ptr, sizeof(DescriptorImpl), stream_));
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
  using typename base_type::DATA_T;
  using typename base_type::DISTANCE_T;
  using typename base_type::INDEX_T;
  using typename base_type::ws_handle;

  const DATA_T* ptr;
  size_t ld;
  uint32_t smem_query_buffer_length;

  _RAFT_HOST_DEVICE standard_dataset_descriptor_t(const DATA_T* ptr,
                                                  INDEX_T size,
                                                  uint32_t dim,
                                                  size_t ld)
    : base_type(size, dim),
      ptr(ptr),
      ld(ld),
      smem_query_buffer_length{raft::round_up_safe<uint32_t>(dim, DatasetBlockDim)}
  {
  }

  _RAFT_HOST_DEVICE [[nodiscard]] auto smem_ws_size_in_bytes() const -> uint32_t
  {
    return smem_query_buffer_length * sizeof(QUERY_T);
  }

  _RAFT_DEVICE [[nodiscard]] auto set_smem_ws(void* smem_ptr) const -> ws_handle
  {
    return reinterpret_cast<ws_handle>(smem_ptr);
  }

  _RAFT_DEVICE void copy_query(ws_handle smem_workspace, const DATA_T* query_ptr) const
  {
    auto buf = smem_query_buffer(smem_workspace);
    for (unsigned i = threadIdx.x; i < smem_query_buffer_length; i += blockDim.x) {
      unsigned j = device::swizzling(i);
      if (i < dim) {
        buf[j] = cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(query_ptr[i]);
      } else {
        buf[j] = 0.0;
      }
    }
  }

  _RAFT_DEVICE void compute_distance_to_random_nodes(
    ws_handle smem_workspace,
    INDEX_T* const result_indices_ptr,       // [num_pickup]
    DISTANCE_T* const result_distances_ptr,  // [num_pickup]
    const size_t num_pickup,
    const unsigned num_distilation,
    const uint64_t rand_xor_mask,
    const INDEX_T* const seed_ptr,  // [num_seeds]
    const uint32_t num_seeds,
    INDEX_T* const visited_hash_ptr,
    const uint32_t hash_bitlen,
    const cuvs::distance::DistanceType metric,
    const uint32_t block_id   = 0,
    const uint32_t num_blocks = 1) const
  {
    return device::compute_distance_to_random_nodes<TeamSize, DatasetBlockDim>(result_indices_ptr,
                                                                               result_distances_ptr,
                                                                               smem_workspace,
                                                                               *this,
                                                                               num_pickup,
                                                                               num_distilation,
                                                                               rand_xor_mask,
                                                                               seed_ptr,
                                                                               num_seeds,
                                                                               visited_hash_ptr,
                                                                               hash_bitlen,
                                                                               metric,
                                                                               block_id,
                                                                               num_blocks);
  }

  _RAFT_DEVICE void compute_distance_to_child_nodes(ws_handle smem_workspace,
                                                    INDEX_T* const result_child_indices_ptr,
                                                    DISTANCE_T* const result_child_distances_ptr,
                                                    // [knn_k, dataset_size]
                                                    const INDEX_T* const knn_graph,
                                                    const uint32_t knn_k,
                                                    // hashmap
                                                    INDEX_T* const visited_hashmap_ptr,
                                                    const uint32_t hash_bitlen,
                                                    const INDEX_T* const parent_indices,
                                                    const INDEX_T* const internal_topk_list,
                                                    const uint32_t search_width,
                                                    const cuvs::distance::DistanceType metric) const
  {
    return device::compute_distance_to_child_nodes<TeamSize, DatasetBlockDim>(
      result_child_indices_ptr,
      result_child_distances_ptr,
      smem_workspace,
      *this,
      knn_graph,
      knn_k,
      visited_hashmap_ptr,
      hash_bitlen,
      parent_indices,
      internal_topk_list,
      search_width,
      metric);
  }

  template <typename T, cuvs::distance::DistanceType METRIC>
  RAFT_DEVICE_INLINE_FUNCTION auto dist_op(T a, T b) const
    -> std::enable_if_t<METRIC == cuvs::distance::DistanceType::L2Expanded, T>
  {
    T diff = a - b;
    return diff * diff;
  }

  template <typename T, cuvs::distance::DistanceType METRIC>
  RAFT_DEVICE_INLINE_FUNCTION auto dist_op(T a, T b) const
    -> std::enable_if_t<METRIC == cuvs::distance::DistanceType::InnerProduct, T>
  {
    return -a * b;
  }

  template <cuvs::distance::DistanceType METRIC>
  RAFT_DEVICE_INLINE_FUNCTION auto compute_similarity(ws_handle smem_workspace,
                                                      const INDEX_T dataset_i,
                                                      const bool valid) const -> DISTANCE_T
  {
    auto query_ptr          = smem_query_buffer(smem_workspace);
    const auto dataset_ptr  = ptr + dataset_i * ld;
    const unsigned lane_id  = threadIdx.x % TeamSize;
    constexpr unsigned vlen = device::get_vlen<LOAD_T, DATA_T>();
    // #include <raft/util/cuda_dev_essentials.cuh
    constexpr unsigned reg_nelem = raft::ceildiv<unsigned>(DatasetBlockDim, TeamSize * vlen);
    raft::TxN_t<DATA_T, vlen> dl_buff[reg_nelem];

    DISTANCE_T norm2 = 0;
    if (valid) {
      for (uint32_t elem_offset = 0; elem_offset < dim; elem_offset += DatasetBlockDim) {
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
    for (uint32_t offset = TeamSize / 2; offset > 0; offset >>= 1) {
      norm2 += __shfl_xor_sync(0xffffffff, norm2, offset);
    }
    return norm2;
  }

 private:
  RAFT_DEVICE_INLINE_FUNCTION constexpr auto smem_query_buffer(ws_handle smem_workspace) const
    -> QUERY_T*
  {
    return reinterpret_cast<QUERY_T*>(smem_workspace);
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
  dataset_descriptor_host<DataT, IndexT, DistanceT> result{
    dd_host, stream, TeamSize, DatasetBlockDim};
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
