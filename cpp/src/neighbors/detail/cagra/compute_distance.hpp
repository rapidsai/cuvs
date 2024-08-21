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

}  // namespace cuvs::neighbors::cagra::detail
