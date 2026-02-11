/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ann_utils.cuh"
#include "cagra/device_common.hpp"
#include "nn_descent_gnnd.hpp"

#include "../../core/omp_wrapper.hpp"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/util/arch.cuh>  // raft::util::arch::SM_*
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <mma.h>

#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <type_traits>

namespace cuvs::neighbors::nn_descent::detail {

template <typename index_t>
struct ResultItem;

template <>
class ResultItem<int> {
 private:
  using index_t = int;
  index_t id_;
  dist_data_t dist_;

 public:
  __host__ __device__ ResultItem()
    : id_(std::numeric_limits<index_t>::max()), dist_(std::numeric_limits<dist_data_t>::max()) {};
  __host__ __device__ ResultItem(const index_t id_with_flag, const dist_data_t dist)
    : id_(id_with_flag), dist_(dist) {};
  __host__ __device__ [[nodiscard]] auto is_new() const -> bool { return id_ >= 0; }
  __host__ __device__ auto id_with_flag() -> index_t& { return id_; }
  __host__ __device__ [[nodiscard]] auto id() const -> index_t
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  __host__ __device__ auto dist() -> dist_data_t& { return dist_; }

  __host__ __device__ void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }

  __host__ __device__ auto operator<(const ResultItem<index_t>& other) const -> bool
  {
    if (dist_ == other.dist_) return id() < other.id();
    return dist_ < other.dist_;
  }
  __host__ __device__ auto operator==(const ResultItem<index_t>& other) const -> bool
  {
    return id() == other.id();
  }
  __host__ __device__ auto operator>=(const ResultItem<index_t>& other) const -> bool
  {
    return !(*this < other);
  }
  __host__ __device__ auto operator<=(const ResultItem<index_t>& other) const -> bool
  {
    return (*this == other) || (*this < other);
  }
  __host__ __device__ auto operator>(const ResultItem<index_t>& other) const -> bool
  {
    return !(*this <= other);
  }
  __host__ __device__ auto operator!=(const ResultItem<index_t>& other) const -> bool
  {
    return !(*this == other);
  }
};

using align32 = raft::Pow2<32>;

template <typename T>
auto get_batch_size(const int it_now, const T nrow, const int batch_size) -> int
{
  int it_total = raft::ceildiv(nrow, batch_size);
  return (it_now == it_total - 1) ? nrow - it_now * batch_size : batch_size;
}

// for avoiding bank conflict
template <typename T>
constexpr __host__ __device__ __forceinline__ auto skew_dim(int ndim) -> int
{
  // all "4"s are for alignment
  if constexpr (std::is_same_v<T, float>) {
    ndim = raft::ceildiv(ndim, 4) * 4;
    return ndim + (ndim % 32 == 0) * 4;
  }
}

template <typename T>
__device__ __forceinline__ auto xor_swap(ResultItem<T> x, int mask, int dir) -> ResultItem<T>
{
  ResultItem<T> y;
  y.dist() = __shfl_xor_sync(raft::warp_full_mask(), x.dist(), mask, raft::warp_size());
  y.id_with_flag() =
    __shfl_xor_sync(raft::warp_full_mask(), x.id_with_flag(), mask, raft::warp_size());
  return x < y == dir ? y : x;
}

__device__ __forceinline__ auto xor_swap(int x, int mask, int dir) -> int
{
  int y = __shfl_xor_sync(raft::warp_full_mask(), x, mask, raft::warp_size());
  return x < y == dir ? y : x;
}

// TODO(snanditale): Move to RAFT utils https://github.com/rapidsai/raft/issues/1827
__device__ __forceinline__ auto bfe(uint lane_id, uint pos) -> uint
{
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

template <typename T>
__device__ __forceinline__ void warp_bitonic_sort(T* element_ptr, const int lane_id)
{
  static_assert(raft::warp_size() == 32);
  auto& element = *element_ptr;
  element       = xor_swap(element, 0x01, bfe(lane_id, 1) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x02, bfe(lane_id, 2) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 2) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x04, bfe(lane_id, 3) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 3) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 3) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x08, bfe(lane_id, 4) ^ bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 4) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 4) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 4) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x10, bfe(lane_id, 4));
  element       = xor_swap(element, 0x08, bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 0));
  return;
}

constexpr int kNumSamples = 32;
// For now, the max. number of samples is 32, so the sample cache size is fixed
// to 64 (32 * 2).
constexpr int kMaxNumBiSamples       = 64;
constexpr int kSkewedMaxNumBiSamples = skew_dim<float>(kMaxNumBiSamples);
constexpr int kBlockSize             = 512;
constexpr int kWmmaM                 = 16;
constexpr int kWmmaN                 = 16;
constexpr int kWmmaK                 = 16;

template <typename DataT>
__device__ __forceinline__ void load_vec(DataT* vec_buffer,
                                         const DataT* d_vec,
                                         const int load_dims,
                                         const int padding_dims,
                                         const int lane_id)
{
  if constexpr (std::is_same_v<DataT, float> or std::is_same_v<DataT, uint8_t> or
                std::is_same_v<DataT, int8_t>) {
    constexpr int kNumLoadElemsPerWarp = raft::warp_size();
    for (int step = 0; step < raft::ceildiv(padding_dims, kNumLoadElemsPerWarp); step++) {
      int idx = step * kNumLoadElemsPerWarp + lane_id;
      if (idx < load_dims) {
        vec_buffer[idx] = d_vec[idx];
      } else if (idx < padding_dims) {
        vec_buffer[idx] = 0.0f;
      }
    }
  }
  if constexpr (std::is_same_v<DataT, __half>) {
    if ((size_t)d_vec % sizeof(float2) == 0 && (size_t)vec_buffer % sizeof(float2) == 0 &&
        load_dims % 4 == 0 && padding_dims % 4 == 0) {
      constexpr int kNumLoadElemsPerWarp = raft::warp_size() * 4;
#pragma unroll
      for (int step = 0; step < raft::ceildiv(padding_dims, kNumLoadElemsPerWarp); step++) {
        int idx_in_vec = step * kNumLoadElemsPerWarp + lane_id * 4;
        if (idx_in_vec + 4 <= load_dims) {
          *reinterpret_cast<float2*>(vec_buffer + idx_in_vec) =
            *reinterpret_cast<const float2*>(d_vec + idx_in_vec);
        } else if (idx_in_vec + 4 <= padding_dims) {
          *reinterpret_cast<float2*>(vec_buffer + idx_in_vec) = float2({.x = 0.0f, .y = 0.0f});
        }
      }
    } else {
      constexpr int kNumLoadElemsPerWarp = raft::warp_size();
      for (int step = 0; step < raft::ceildiv(padding_dims, kNumLoadElemsPerWarp); step++) {
        int idx = step * kNumLoadElemsPerWarp + lane_id;
        if (idx < load_dims) {
          vec_buffer[idx] = d_vec[idx];
        } else if (idx < padding_dims) {
          vec_buffer[idx] = 0.0f;
        }
      }
    }
  }
}

// TODO(snanditale): Replace with RAFT utilities https://github.com/rapidsai/raft/issues/1827
/** Calculate L2 norm, and cast data to OutputT */
template <typename DataT, typename OutputT = __half>
RAFT_KERNEL preprocess_data_kernel(
  const DataT* input_data,
  OutputT* output_data,
  int dim,
  dist_data_t* l2_norms,
  size_t list_offset                  = 0,
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
{
  extern __shared__ char buffer[];
  __shared__ float l2_norm;
  auto* s_vec    = reinterpret_cast<DataT*>(buffer);
  size_t list_id = list_offset + blockIdx.x;

  load_vec(s_vec,
           input_data + static_cast<size_t>(blockIdx.x) * dim,
           dim,
           dim,
           threadIdx.x % raft::warp_size());
  if (threadIdx.x == 0) { l2_norm = 0; }
  __syncthreads();

  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      metric == cuvs::distance::DistanceType::CosineExpanded) {
    int lane_id = threadIdx.x % raft::warp_size();
    for (int step = 0; step < raft::ceildiv(dim, raft::warp_size()); step++) {
      int idx         = step * raft::warp_size() + lane_id;
      float part_dist = 0;
      if (idx < dim) {
        part_dist = s_vec[idx];
        part_dist = part_dist * part_dist;
      }
      __syncwarp();
      for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
        part_dist += __shfl_down_sync(raft::warp_full_mask(), part_dist, offset);
      }
      if (lane_id == 0) { l2_norm += part_dist; }
      __syncwarp();
    }
  }

  for (int step = 0; step < raft::ceildiv(dim, raft::warp_size()); step++) {
    int idx = step * raft::warp_size() + threadIdx.x;
    if (idx < dim) {
      if (metric == cuvs::distance::DistanceType::InnerProduct) {
        output_data[list_id * dim + idx] = input_data[(size_t)blockIdx.x * dim + idx];
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        output_data[list_id * dim + idx] =
          (float)input_data[(size_t)blockIdx.x * dim + idx] / sqrt(l2_norm);
      } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        int idx_for_byte           = list_id * dim + idx;  // uint8 or int8 data
        auto* output_bytes         = reinterpret_cast<uint8_t*>(output_data);
        output_bytes[idx_for_byte] = input_data[(size_t)blockIdx.x * dim + idx];
      } else {  // L2Expanded or L2SqrtExpanded
        output_data[list_id * dim + idx] = input_data[(size_t)blockIdx.x * dim + idx];
        if (idx == 0) { l2_norms[list_id] = l2_norm; }
      }
    }
  }
}

template <typename index_t>
RAFT_KERNEL add_rev_edges_kernel(const index_t* graph,
                                 index_t* rev_graph,
                                 int num_samples,
                                 int2* list_sizes)
{
  size_t list_id = blockIdx.x;
  int2 list_size = list_sizes[list_id];

  for (int idx = threadIdx.x; idx < list_size.x; idx += blockDim.x) {
    // each node has same number (num_samples) of forward and reverse edges
    index_t rev_list_id = graph[list_id * num_samples + idx];
    if (rev_list_id == std::numeric_limits<index_t>::max()) {
      // sentinel value
      continue;
    }

    // there are already num_samples forward edges
    int idx_in_rev_list = atomicAdd(&list_sizes[rev_list_id].y, 1);
    if (idx_in_rev_list >= num_samples) {
      atomicExch(&list_sizes[rev_list_id].y, num_samples);
    } else {
      rev_graph[rev_list_id * num_samples + idx_in_rev_list] = list_id;
    }
  }
}

template <typename index_t, typename IdT = InternalID_t<index_t>>
__device__ void insert_to_global_graph(ResultItem<index_t> elem,
                                       size_t list_id,
                                       IdT* graph,
                                       dist_data_t* dists,
                                       int node_degree,
                                       int* locks)
{
  int tx                 = threadIdx.x;
  int lane_id            = tx % raft::warp_size();
  size_t global_idx_base = list_id * node_degree;
  if (elem.id() == list_id) return;

  const int num_segments = raft::ceildiv(node_degree, raft::warp_size());

  int loop_flag = 0;
  do {
    int segment_id = elem.id() % num_segments;
    if (lane_id == 0) {
      loop_flag = atomicCAS(&locks[list_id * num_segments + segment_id], 0, 1) == 0;
    }

    loop_flag = __shfl_sync(raft::warp_full_mask(), loop_flag, 0);

    if (loop_flag == 1) {
      ResultItem<index_t> knn_list_frag;
      int local_idx     = segment_id * raft::warp_size() + lane_id;
      size_t global_idx = global_idx_base + local_idx;
      if (local_idx < node_degree) {
        knn_list_frag.id_with_flag() = graph[global_idx].id_with_flag();
        knn_list_frag.dist()         = dists[global_idx];
      }

      int pos_to_insert = -1;
      ResultItem<index_t> prev_elem;

      prev_elem.id_with_flag() =
        __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.id_with_flag(), 1);
      prev_elem.dist() = __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.dist(), 1);

      if (lane_id == 0) {
        prev_elem = ResultItem<index_t>{std::numeric_limits<index_t>::min(),
                                        std::numeric_limits<dist_data_t>::lowest()};
      }
      if (elem > prev_elem && elem < knn_list_frag) {
        pos_to_insert = segment_id * raft::warp_size() + lane_id;
      } else if (elem == prev_elem || elem == knn_list_frag) {
        pos_to_insert = -2;
      }
      uint mask = __ballot_sync(raft::warp_full_mask(), pos_to_insert >= 0);
      if (mask) {
        uint set_lane_id = __fns(mask, 0, 1);
        pos_to_insert    = __shfl_sync(raft::warp_full_mask(), pos_to_insert, set_lane_id);
      }

      if (pos_to_insert >= 0) {
        int local_idx = segment_id * raft::warp_size() + lane_id;
        if (local_idx > pos_to_insert) {
          local_idx++;
        } else if (local_idx == pos_to_insert) {
          graph[global_idx_base + local_idx].id_with_flag() = elem.id_with_flag();
          dists[global_idx_base + local_idx]                = elem.dist();
          local_idx++;
        }
        size_t global_pos = global_idx_base + local_idx;
        if (local_idx < (segment_id + 1) * raft::warp_size() && local_idx < node_degree) {
          graph[global_pos].id_with_flag() = knn_list_frag.id_with_flag();
          dists[global_pos]                = knn_list_frag.dist();
        }
      }
      __threadfence();
      if (loop_flag && lane_id == 0) { atomicExch(&locks[list_id * num_segments + segment_id], 0); }
    }
  } while (!loop_flag);
}

template <typename index_t>
__device__ auto get_min_item(const index_t id,
                             const int idx_in_list,
                             const index_t* neighbs,
                             const dist_data_t* distances,
                             const bool find_in_row = true) -> ResultItem<index_t>
{
  int lane_id = threadIdx.x % raft::warp_size();

  static_assert(kMaxNumBiSamples == 64);
  int idx[kMaxNumBiSamples / raft::warp_size()];
  float dist[kMaxNumBiSamples / raft::warp_size()] = {std::numeric_limits<dist_data_t>::max(),
                                                      std::numeric_limits<dist_data_t>::max()};
  idx[0]                                           = lane_id;
  idx[1]                                           = raft::warp_size() + lane_id;

  if (neighbs[idx[0]] != id) {
    dist[0] = find_in_row ? distances[idx_in_list * kSkewedMaxNumBiSamples + lane_id]
                          : distances[idx_in_list + lane_id * kSkewedMaxNumBiSamples];
  }

  if (neighbs[idx[1]] != id) {
    dist[1] = find_in_row
                ? distances[idx_in_list * kSkewedMaxNumBiSamples + raft::warp_size() + lane_id]
                : distances[idx_in_list + (raft::warp_size() + lane_id) * kSkewedMaxNumBiSamples];
  }

  if (dist[1] < dist[0]) {
    dist[0] = dist[1];
    idx[0]  = idx[1];
  }
  __syncwarp();
  for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
    float other_idx  = __shfl_down_sync(raft::warp_full_mask(), idx[0], offset);
    float other_dist = __shfl_down_sync(raft::warp_full_mask(), dist[0], offset);
    if (other_dist < dist[0]) {
      dist[0] = other_dist;
      idx[0]  = other_idx;
    }
  }

  ResultItem<index_t> result;
  result.dist()         = __shfl_sync(raft::warp_full_mask(), dist[0], 0);
  result.id_with_flag() = neighbs[__shfl_sync(raft::warp_full_mask(), idx[0], 0)];
  return result;
}

template <typename T>
__device__ __forceinline__ void remove_duplicates(
  T* list_a, int list_a_size, T* list_b, int list_b_size, int& unique_counter, int execute_warp_id)
{
  static_assert(raft::warp_size() == 32);
  if (!(threadIdx.x >= execute_warp_id * raft::warp_size() &&
        threadIdx.x < execute_warp_id * raft::warp_size() + raft::warp_size())) {
    return;
  }
  int lane_id = threadIdx.x % raft::warp_size();
  T elem      = std::numeric_limits<T>::max();
  if (lane_id < list_a_size) { elem = list_a[lane_id]; }
  warp_bitonic_sort(&elem, lane_id);

  if (elem != std::numeric_limits<T>::max()) { list_a[lane_id] = elem; }

  T elem_b = std::numeric_limits<T>::max();

  if (lane_id < list_b_size) { elem_b = list_b[lane_id]; }
  __syncwarp();

  int idx_l    = 0;
  int idx_r    = list_a_size;
  bool existed = false;
  while (idx_l < idx_r) {
    int idx  = (idx_l + idx_r) / 2;
    int elem = list_a[idx];
    if (elem == elem_b) {
      existed = true;
      break;
    }
    if (elem_b > elem) {
      idx_l = idx + 1;
    } else {
      idx_r = idx;
    }
  }
  if (!existed && elem_b != std::numeric_limits<T>::max()) {
    int idx                   = atomicAdd(&unique_counter, 1);
    list_a[list_a_size + idx] = elem_b;
  }
}

template <typename index_t, typename DataT, typename DistEpilogueT>
__device__ __forceinline__ void calculate_metric(float* s_distances,
                                                 index_t* row_neighbors,
                                                 int list_row_size,
                                                 index_t* col_neighbors,
                                                 int list_col_size,
                                                 const DataT* data,
                                                 const int data_dim,
                                                 dist_data_t* l2_norms,
                                                 cuvs::distance::DistanceType metric,
                                                 DistEpilogueT dist_epilogue)
{
  // if we have a distance epilogue, distances need to be fully calculated instead of postprocessing
  // them.
  bool can_postprocess_dist = std::is_same_v<DistEpilogueT, raft::identity_op>;

  for (int i = threadIdx.x; i < kMaxNumBiSamples * kSkewedMaxNumBiSamples; i += blockDim.x) {
    int row_id = i / kSkewedMaxNumBiSamples;
    int col_id = i % kSkewedMaxNumBiSamples;

    if (row_id < list_row_size && col_id < list_col_size) {
      if (metric == cuvs::distance::DistanceType::InnerProduct && can_postprocess_dist) {
        s_distances[i] = -s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        s_distances[i] = 1.0 - s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        s_distances[i] = 0.0;
        int n1         = row_neighbors[row_id];
        int n2         = col_neighbors[col_id];
        // TODO(snanditale): https://github.com/rapidsai/cuvs/issues/1127
        const uint8_t* data_n1 = reinterpret_cast<const uint8_t*>(data) + n1 * data_dim;
        const uint8_t* data_n2 = reinterpret_cast<const uint8_t*>(data) + n2 * data_dim;
        for (int d = 0; d < data_dim; d++) {
          s_distances[i] += __popc(static_cast<uint32_t>(data_n1[d] ^ data_n2[d]) & 0xff);
        }
      } else {  // L2Expanded or L2SqrtExpanded
        s_distances[i] =
          l2_norms[row_neighbors[row_id]] + l2_norms[col_neighbors[col_id]] - 2.0 * s_distances[i];
        // for fp32 vs fp16 precision differences resulting in negative distances when distance
        // should be 0 related issue: https://github.com/rapidsai/cuvs/issues/991
        s_distances[i] = s_distances[i] < 0.0f ? 0.0f : s_distances[i];
        if (!can_postprocess_dist && metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          s_distances[i] = sqrtf(s_distances[i]);
        }
      }
      s_distances[i] = dist_epilogue(s_distances[i], row_neighbors[row_id], col_neighbors[col_id]);
    } else {
      s_distances[i] = std::numeric_limits<float>::max();
    }
  }
}

// launch_bounds here denote kBlockSize = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = kBlockSize * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore
template <typename index_t, typename IdT = InternalID_t<index_t>, typename DistEpilogueT>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
// Use minBlocksPerMultiprocessor = 4 on specific arches
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(kBlockSize, 4)
#else
__launch_bounds__(kBlockSize)  // NOLINT(readability-identifier-naming)
#endif
#endif
  local_join_kernel(const index_t* graph_new,
                    const index_t* rev_graph_new,
                    const int2* sizes_new,
                    const index_t* graph_old,
                    const index_t* rev_graph_old,
                    const int2* sizes_old,
                    const int width,
                    const float* data,
                    const int data_dim,
                    IdT* graph,
                    dist_data_t* dists,
                    int graph_width,
                    int* locks,
                    dist_data_t* l2_norms,
                    cuvs::distance::DistanceType metric,
                    DistEpilogueT dist_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  namespace wmma = nvcuda::wmma;
  __shared__ int s_list[kMaxNumBiSamples * 2];

  constexpr int APAD           = 4;
  constexpr int BPAD           = 4;
  constexpr int TILE_COL_WIDTH = 32;
  __shared__ float s_nv[kMaxNumBiSamples][TILE_COL_WIDTH + APAD];
  __shared__ float s_ov[kMaxNumBiSamples][TILE_COL_WIDTH + BPAD];
  __shared__ float s_distances[kMaxNumBiSamples * kSkewedMaxNumBiSamples];

  // s_distances: kMaxNumBiSamples x kSkewedMaxNumBiSamples, reuse the space of s_ov
  int* s_unique_counter = reinterpret_cast<int*>(&s_ov[0][0]);

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  index_t* new_neighbors = s_list;
  index_t* old_neighbors = s_list + kMaxNumBiSamples;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = kBlockSize / raft::warp_size();

  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    int tid = threadIdx.x;
    for (int i = tid; i < kMaxNumBiSamples * kSkewedMaxNumBiSamples; i += blockDim.x)
      s_distances[i] = 0.0f;

    __syncthreads();

    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
#pragma unroll
      for (int i = 0; i < kMaxNumBiSamples / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      // this is much faster than a warp-collaborative multiplication because kMaxNumBiSamples is
      // fixed and small (64)
      for (int i = threadIdx.x; i < kMaxNumBiSamples * kSkewedMaxNumBiSamples; i += blockDim.x) {
        int tmp_row = i / kSkewedMaxNumBiSamples;
        int tmp_col = i % kSkewedMaxNumBiSamples;
        if (tmp_row < list_new_size && tmp_col < list_new_size) {
          float acc = 0.0f;
          for (int d = 0; d < num_load_elems; d++) {
            acc += s_nv[tmp_row][d] * s_nv[tmp_col][d];
          }
          s_distances[i] += acc;
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   new_neighbors,
                   list_new_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size) continue;
    auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();

  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    int tid = threadIdx.x;
    for (int i = tid; i < kMaxNumBiSamples * kSkewedMaxNumBiSamples; i += blockDim.x)
      s_distances[i] = 0.0f;

    __syncthreads();

    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
      if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
        for (int i = 0; i < kMaxNumBiSamples / num_warps; i++) {
          int idx = i * num_warps + warp_id;
          if (idx < list_new_size) {
            size_t neighbor_id = new_neighbors[idx];
            size_t idx_in_data = neighbor_id * data_dim;
            load_vec(s_nv[idx],
                     data + idx_in_data + step * TILE_COL_WIDTH,
                     num_load_elems,
                     TILE_COL_WIDTH,
                     lane_id);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kMaxNumBiSamples / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_old_size) {
          size_t neighbor_id = old_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_ov[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      // this is much faster than a warp-collaborative multiplication because kMaxNumBiSamples is
      // fixed and small (64)
      for (int i = threadIdx.x; i < kMaxNumBiSamples * kSkewedMaxNumBiSamples; i += blockDim.x) {
        int tmp_row = i / kSkewedMaxNumBiSamples;
        int tmp_col = i % kSkewedMaxNumBiSamples;
        if (tmp_row < list_new_size && tmp_col < list_old_size) {
          float acc = 0.0f;
          for (int d = 0; d < num_load_elems; d++) {
            acc += s_nv[tmp_row][d] * s_ov[tmp_col][d];
          }
          s_distances[i] += acc;
        }
      }
      __syncthreads();
    }
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   old_neighbors,
                   list_old_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(kMaxNumBiSamples * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size && idx_in_list < kMaxNumBiSamples) continue;
    if (idx_in_list >= kMaxNumBiSamples + list_old_size && idx_in_list < kMaxNumBiSamples * 2)
      continue;
    ResultItem<index_t> min_elem{std::numeric_limits<index_t>::max(),
                                 std::numeric_limits<dist_data_t>::max()};
    if (idx_in_list < kMaxNumBiSamples) {
      auto temp_min_item =
        get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item(
        s_list[idx_in_list], idx_in_list - kMaxNumBiSamples, new_neighbors, s_distances, false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }

    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif
}

// launch_bounds here denote kBlockSize = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = kBlockSize * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore
template <typename index_t, typename IdT = InternalID_t<index_t>, typename DistEpilogueT>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
// Use minBlocksPerMultiprocessor = 4 on specific arches
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(kBlockSize, 4)
#else
__launch_bounds__(kBlockSize)  // NOLINT(readability-identifier-naming)
#endif
#endif
  local_join_kernel(const index_t* graph_new,
                    const index_t* rev_graph_new,
                    const int2* sizes_new,
                    const index_t* graph_old,
                    const index_t* rev_graph_old,
                    const int2* sizes_old,
                    const int width,
                    const __half* data,
                    const int data_dim,
                    IdT* graph,
                    dist_data_t* dists,
                    int graph_width,
                    int* locks,
                    dist_data_t* l2_norms,
                    cuvs::distance::DistanceType metric,
                    DistEpilogueT dist_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  namespace wmma = nvcuda::wmma;
  __shared__ int s_list[kMaxNumBiSamples * 2];

  constexpr int APAD           = 8;
  constexpr int BPAD           = 8;
  constexpr int TILE_COL_WIDTH = 128;
  __shared__ __half s_nv[kMaxNumBiSamples][TILE_COL_WIDTH + APAD];  // New vectors
  __shared__ __half s_ov[kMaxNumBiSamples][TILE_COL_WIDTH + BPAD];  // Old vectors
  static_assert(sizeof(float) * kMaxNumBiSamples * kSkewedMaxNumBiSamples <=
                sizeof(__half) * kMaxNumBiSamples * (TILE_COL_WIDTH + BPAD));
  // s_distances: kMaxNumBiSamples x kSkewedMaxNumBiSamples, reuse the space of s_ov
  float* s_distances    = (float*)&s_ov[0][0];
  int* s_unique_counter = reinterpret_cast<int*>(&s_ov[0][0]);

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  index_t* new_neighbors = s_list;
  index_t* old_neighbors = s_list + kMaxNumBiSamples;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = kBlockSize / raft::warp_size();

  int warp_id_y = warp_id / 4;
  int warp_id_x = warp_id % 4;

  wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_frag;
  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    wmma::fill_fragment(c_frag, 0.0);

    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
#pragma unroll
      for (int i = 0; i < kMaxNumBiSamples / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      for (int i = 0; i < TILE_COL_WIDTH / kWmmaK; i++) {
        wmma::load_matrix_sync(
          a_frag, s_nv[warp_id_y * kWmmaM] + i * kWmmaK, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(
          b_frag, s_nv[warp_id_x * kWmmaN] + i * kWmmaK, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
      }
    }

    wmma::store_matrix_sync(
      s_distances + warp_id_y * kWmmaM * kSkewedMaxNumBiSamples + warp_id_x * kWmmaN,
      c_frag,
      kSkewedMaxNumBiSamples,
      wmma::mem_row_major);
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   new_neighbors,
                   list_new_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);
  __syncthreads();

  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size) continue;
    auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();

  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    wmma::fill_fragment(c_frag, 0.0);
    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
      if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
        for (int i = 0; i < kMaxNumBiSamples / num_warps; i++) {
          int idx = i * num_warps + warp_id;
          if (idx < list_new_size) {
            size_t neighbor_id = new_neighbors[idx];
            size_t idx_in_data = neighbor_id * data_dim;
            load_vec(s_nv[idx],
                     data + idx_in_data + step * TILE_COL_WIDTH,
                     num_load_elems,
                     TILE_COL_WIDTH,
                     lane_id);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < kMaxNumBiSamples / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_old_size) {
          size_t neighbor_id = old_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_ov[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      for (int i = 0; i < TILE_COL_WIDTH / kWmmaK; i++) {
        wmma::load_matrix_sync(
          a_frag, s_nv[warp_id_y * kWmmaM] + i * kWmmaK, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(
          b_frag, s_ov[warp_id_x * kWmmaN] + i * kWmmaK, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
      }
    }

    wmma::store_matrix_sync(
      s_distances + warp_id_y * kWmmaM * kSkewedMaxNumBiSamples + warp_id_x * kWmmaN,
      c_frag,
      kSkewedMaxNumBiSamples,
      wmma::mem_row_major);
    __syncthreads();
  }

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   old_neighbors,
                   list_old_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(kMaxNumBiSamples * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size && idx_in_list < kMaxNumBiSamples) continue;
    if (idx_in_list >= kMaxNumBiSamples + list_old_size && idx_in_list < kMaxNumBiSamples * 2)
      continue;
    ResultItem<index_t> min_elem{std::numeric_limits<index_t>::max(),
                                 std::numeric_limits<dist_data_t>::max()};
    if (idx_in_list < kMaxNumBiSamples) {
      auto temp_min_item =
        get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item(
        s_list[idx_in_list], idx_in_list - kMaxNumBiSamples, new_neighbors, s_distances, false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }

    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif
}

namespace {
template <typename index_t>
auto insert_to_ordered_list(InternalID_t<index_t>* list,
                            dist_data_t* dist_list,
                            const int width,
                            const InternalID_t<index_t> neighb_id,
                            const dist_data_t dist) -> int
{
  if (dist > dist_list[width - 1]) { return width; }

  int idx_insert      = width;
  bool position_found = false;
  for (int i = 0; i < width; i++) {
    if (list[i].id() == neighb_id.id()) { return width; }
    if (!position_found && dist_list[i] > dist) {
      idx_insert     = i;
      position_found = true;
    }
  }
  if (idx_insert == width) return idx_insert;

  memmove(list + idx_insert + 1, list + idx_insert, sizeof(*list) * (width - idx_insert - 1));
  memmove(dist_list + idx_insert + 1,
          dist_list + idx_insert,
          sizeof(*dist_list) * (width - idx_insert - 1));

  list[idx_insert]      = neighb_id;
  dist_list[idx_insert] = dist;
  return idx_insert;
};

}  // namespace

template <typename index_t>
gnnd_graph<index_t>::gnnd_graph(raft::resources const& res,
                                const size_t nrow,
                                const size_t node_degree,
                                const size_t internal_node_degree,
                                const size_t num_samples)
  : res(res),
    nrow(nrow),
    node_degree(node_degree),
    num_samples(num_samples),
    bloom_filter(nrow, internal_node_degree / kSegmentSize, 3),
    h_dists{raft::make_host_matrix<dist_data_t, size_t, raft::row_major>(nrow, node_degree)},
    h_graph_new{raft::make_pinned_matrix<index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    h_list_sizes_new{raft::make_pinned_vector<int2, size_t>(res, nrow)},
    h_graph_old{raft::make_pinned_matrix<index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    h_list_sizes_old{raft::make_pinned_vector<int2, size_t>(res, nrow)}
{
  // node_degree must be a multiple of kSegmentSize;
  assert(node_degree % kSegmentSize == 0);
  assert(internal_node_degree % kSegmentSize == 0);

  num_segments = node_degree / kSegmentSize;
  // To save the CPU memory, graph should be allocated by external function
  h_graph = nullptr;
}

// This is the only operation on the CPU that cannot be overlapped.
// So it should be as fast as possible.
template <typename index_t>
void gnnd_graph<index_t>::sample_graph_new(InternalID_t<index_t>* new_neighbors, const size_t width)
{
  std::fill_n(h_graph_new.data_handle(), nrow * num_samples, std::numeric_limits<index_t>::max());
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    auto list_new                       = h_graph_new.data_handle() + i * num_samples;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighb_id >= nrow) break;
      if (bloom_filter.check(i, new_neighb_id)) { continue; }
      bloom_filter.add(i, new_neighb_id);
      new_neighbors[i * width + j].mark_old();
      list_new[h_list_sizes_new.data_handle()[i].x++] = new_neighb_id;
      if (h_list_sizes_new.data_handle()[i].x == num_samples) break;
    }
  }
}

template <typename index_t>
void gnnd_graph<index_t>::init_random_graph()
{
  for (size_t seg_idx = 0; seg_idx < static_cast<size_t>(num_segments); seg_idx++) {
    // random sequence (range: 0~nrow)
    // segment_x stores neighbors which id % num_segments == x
    std::vector<index_t> rand_seq((nrow + num_segments - 1) / num_segments);
    std::iota(rand_seq.begin(), rand_seq.end(), 0);
    auto gen = std::default_random_engine{seg_idx};
    std::shuffle(rand_seq.begin(), rand_seq.end(), gen);

#pragma omp parallel for
    for (size_t i = 0; i < nrow; i++) {
      size_t base_idx         = i * node_degree + seg_idx * kSegmentSize;
      auto h_neighbor_list    = h_graph + base_idx;
      auto h_dist_list        = h_dists.data_handle() + base_idx;
      size_t idx              = base_idx;
      size_t self_in_this_seg = 0;
      for (size_t j = 0; j < static_cast<size_t>(kSegmentSize); j++) {
        index_t id = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;
        if ((size_t)id == i) {
          idx++;
          id               = rand_seq[idx % rand_seq.size()] * num_segments + seg_idx;
          self_in_this_seg = 1;
        }

        h_neighbor_list[j].id_with_flag() =
          j < (rand_seq.size() - self_in_this_seg) && size_t(id) < nrow
            ? id
            : std::numeric_limits<index_t>::max();
        h_dist_list[j] = std::numeric_limits<dist_data_t>::max();
        idx++;
      }
    }
  }
}

template <typename index_t>
void gnnd_graph<index_t>::sample_graph(bool sample_new)
{
  std::fill_n(h_graph_old.data_handle(), nrow * num_samples, std::numeric_limits<index_t>::max());
  if (sample_new) {
    std::fill_n(h_graph_new.data_handle(), nrow * num_samples, std::numeric_limits<index_t>::max());
  }

#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    h_list_sizes_old.data_handle()[i].x = 0;
    h_list_sizes_old.data_handle()[i].y = 0;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list     = h_graph + i * node_degree;
    auto list_old = h_graph_old.data_handle() + i * num_samples;
    auto list_new = h_graph_new.data_handle() + i * num_samples;
    for (int j = 0; j < kSegmentSize; j++) {
      for (int k = 0; k < num_segments; k++) {
        auto neighbor = list[k * kSegmentSize + j];
        if ((size_t)neighbor.id() >= nrow) continue;
        if (!neighbor.is_new()) {
          if (h_list_sizes_old.data_handle()[i].x < num_samples) {
            list_old[h_list_sizes_old.data_handle()[i].x++] = neighbor.id();
          }
        } else if (sample_new) {
          if (h_list_sizes_new.data_handle()[i].x < num_samples) {
            list[k * kSegmentSize + j].mark_old();
            list_new[h_list_sizes_new.data_handle()[i].x++] = neighbor.id();
          }
        }
        if (h_list_sizes_old.data_handle()[i].x == num_samples &&
            h_list_sizes_new.data_handle()[i].x == num_samples) {
          break;
        }
      }
      if (h_list_sizes_old.data_handle()[i].x == num_samples &&
          h_list_sizes_new.data_handle()[i].x == num_samples) {
        break;
      }
    }
  }
}

template <typename index_t>
void gnnd_graph<index_t>::update_graph(const InternalID_t<index_t>* new_neighbors,
                                       const dist_data_t* new_dists,
                                       const size_t width,
                                       std::atomic<int64_t>& update_counter)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j];
      auto new_dist      = new_dists[i * width + j];
      if (new_dist == std::numeric_limits<dist_data_t>::max()) break;
      if ((size_t)new_neighb_id.id() == i) continue;
      int seg_idx    = new_neighb_id.id() % num_segments;
      auto list      = h_graph + i * node_degree + seg_idx * kSegmentSize;
      auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * kSegmentSize;
      int insert_pos =
        insert_to_ordered_list(list, dist_list, kSegmentSize, new_neighb_id, new_dist);
      if (i % kCounterInterval == 0 && insert_pos != kSegmentSize) { update_counter++; }
    }
  }
}

template <typename index_t>
void gnnd_graph<index_t>::sort_lists()
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    std::vector<std::pair<dist_data_t, index_t>> new_list;
    for (size_t j = 0; j < node_degree; j++) {
      new_list.emplace_back(h_dists.data_handle()[i * node_degree + j],
                            h_graph[i * node_degree + j].id());
    }
    std::sort(new_list.begin(), new_list.end());
    for (size_t j = 0; j < node_degree; j++) {
      h_graph[i * node_degree + j].id_with_flag() = new_list[j].second;
      h_dists.data_handle()[i * node_degree + j]  = new_list[j].first;
    }
  }
}

template <typename index_t>
void gnnd_graph<index_t>::clear()
{
  bloom_filter.clear();
}

template <typename index_t>
gnnd_graph<index_t>::~gnnd_graph()
{
  assert(h_graph == nullptr);
}

template <typename DataT, typename index_t>
gnnd<DataT, index_t>::gnnd(raft::resources const& res, const build_config& build_config)
  : res_(res),
    build_config_(build_config),
    graph_(res,
           build_config.max_dataset_size,
           align32::roundUp(build_config.node_degree),
           align32::roundUp(build_config.internal_node_degree ? build_config.internal_node_degree
                                                              : build_config.node_degree),
           kNumSamples),
    nrow_(build_config.max_dataset_size),
    ndim_(build_config.dataset_dim),
    l2_norms_{raft::make_device_vector<dist_data_t, size_t>(res, 0)},
    graph_buffer_{
      raft::make_device_matrix<id_t, size_t, raft::row_major>(res, nrow_, kDegreeOnDevice)},
    dists_buffer_{
      raft::make_device_matrix<dist_data_t, size_t, raft::row_major>(res, nrow_, kDegreeOnDevice)},
    graph_host_buffer_{
      raft::make_pinned_matrix<id_t, size_t, raft::row_major>(res, nrow_, kDegreeOnDevice)},
    dists_host_buffer_{
      raft::make_pinned_matrix<dist_data_t, size_t, raft::row_major>(res, nrow_, kDegreeOnDevice)},
    d_locks_{raft::make_device_vector<int, size_t>(res, nrow_)},
    h_rev_graph_new_{
      raft::make_pinned_matrix<index_t, size_t, raft::row_major>(res, nrow_, kNumSamples)},
    h_graph_old_(
      raft::make_pinned_matrix<index_t, size_t, raft::row_major>(res, nrow_, kNumSamples)),
    h_rev_graph_old_{
      raft::make_pinned_matrix<index_t, size_t, raft::row_major>(res, nrow_, kNumSamples)},
    d_list_sizes_new_{raft::make_device_vector<int2, size_t>(res, nrow_)},
    d_list_sizes_old_{raft::make_device_vector<int2, size_t>(res, nrow_)}
{
  static_assert(kNumSamples <= 32);

  using input_t = std::remove_const_t<DataT>;
  if (std::is_same_v<input_t, float> &&
      (build_config.dist_comp_dtype == cuvs::neighbors::nn_descent::DIST_COMP_DTYPE::FP32 ||
       (build_config.dist_comp_dtype == cuvs::neighbors::nn_descent::DIST_COMP_DTYPE::AUTO &&
        build_config.dataset_dim <= 16))) {
    // use fp32 distance computation for better precision with smaller dimension
    d_data_float_.emplace(
      raft::make_device_matrix<float, size_t, raft::row_major>(res, nrow_, ndim_));
  } else {
    d_data_half_.emplace(raft::make_device_matrix<half, size_t, raft::row_major>(
      res,
      nrow_,
      build_config.metric == cuvs::distance::DistanceType::BitwiseHamming
        ? (build_config.dataset_dim + 1) / 2
        : build_config.dataset_dim));
  }

  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<index_t, int64_t>(
    reinterpret_cast<index_t*>(graph_buffer_.data_handle()), nrow_, kDegreeOnDevice);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);

  if (build_config.metric == cuvs::distance::DistanceType::L2Expanded ||
      build_config.metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    l2_norms_ = raft::make_device_vector<dist_data_t, size_t>(res, nrow_);
  }
};

template <typename DataT, typename index_t>
void gnnd<DataT, index_t>::reset(raft::resources const& res)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<index_t, int64_t>(
    reinterpret_cast<index_t*>(graph_buffer_.data_handle()), nrow_, kDegreeOnDevice);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);
}

template <typename DataT, typename index_t>
void gnnd<DataT, index_t>::add_reverse_edges(index_t* graph_ptr,
                                             index_t* h_rev_graph_ptr,
                                             index_t* d_rev_graph_ptr,
                                             int2* list_sizes,
                                             cudaStream_t stream)
{
  raft::matrix::fill(
    res_,
    raft::make_device_matrix_view<index_t, int64_t>(d_rev_graph_ptr, nrow_, kDegreeOnDevice),
    std::numeric_limits<index_t>::max());
  add_rev_edges_kernel<<<nrow_, raft::warp_size(), 0, stream>>>(
    graph_ptr, d_rev_graph_ptr, kNumSamples, list_sizes);
  raft::copy(h_rev_graph_ptr, d_rev_graph_ptr, nrow_ * kNumSamples, stream);
}

template <typename DataT, typename index_t>
template <typename DistEpilogueT>
void gnnd<DataT, index_t>::local_join(cudaStream_t stream, DistEpilogueT dist_epilogue)
{
  raft::matrix::fill(res_, dists_buffer_.view(), std::numeric_limits<float>::max());
  if (d_data_float_.has_value()) {
    local_join_kernel<<<nrow_, kBlockSize, 0, stream>>>(graph_.h_graph_new.data_handle(),
                                                        h_rev_graph_new_.data_handle(),
                                                        d_list_sizes_new_.data_handle(),
                                                        h_graph_old_.data_handle(),
                                                        h_rev_graph_old_.data_handle(),
                                                        d_list_sizes_old_.data_handle(),
                                                        kNumSamples,
                                                        d_data_float_.value().data_handle(),
                                                        ndim_,
                                                        graph_buffer_.data_handle(),
                                                        dists_buffer_.data_handle(),
                                                        kDegreeOnDevice,
                                                        d_locks_.data_handle(),
                                                        l2_norms_.data_handle(),
                                                        build_config_.metric,
                                                        dist_epilogue);
  } else {
    local_join_kernel<<<nrow_, kBlockSize, 0, stream>>>(graph_.h_graph_new.data_handle(),
                                                        h_rev_graph_new_.data_handle(),
                                                        d_list_sizes_new_.data_handle(),
                                                        h_graph_old_.data_handle(),
                                                        h_rev_graph_old_.data_handle(),
                                                        d_list_sizes_old_.data_handle(),
                                                        kNumSamples,
                                                        d_data_half_.value().data_handle(),
                                                        ndim_,
                                                        graph_buffer_.data_handle(),
                                                        dists_buffer_.data_handle(),
                                                        kDegreeOnDevice,
                                                        d_locks_.data_handle(),
                                                        l2_norms_.data_handle(),
                                                        build_config_.metric,
                                                        dist_epilogue);
  }
}

template <typename DataT, typename index_t>
template <typename DistEpilogueT>
void gnnd<DataT, index_t>::build(DataT* data,
                                 const index_t nrow,
                                 index_t* output_graph,
                                 bool return_distances,
                                 dist_data_t* output_distances,
                                 DistEpilogueT dist_epilogue)
{
  using input_t = std::remove_const_t<DataT>;

  if (build_config_.metric == distance::DistanceType::BitwiseHamming &&
      !(std::is_same_v<input_t, uint8_t> || std::is_same_v<input_t, int8_t>)) {
    RAFT_FAIL(
      "Data type needs to be int8 or uint8 for NN Descent to run with BitwiseHamming distance.");
  }

  cudaStream_t stream = raft::resource::get_cuda_stream(res_);
  nrow_               = nrow;
  graph_.nrow         = nrow;
  graph_.bloom_filter.set_nrow(nrow);
  update_counter_ = 0;
  graph_.h_graph  = reinterpret_cast<InternalID_t<index_t>*>(output_graph);

  if (d_data_float_.has_value()) {
    raft::matrix::fill(res_, d_data_float_.value().view(), static_cast<float>(0));
  } else {
    raft::matrix::fill(res_, d_data_half_.value().view(), static_cast<half>(0));
  }

  cudaPointerAttributes data_ptr_attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&data_ptr_attr, data));
  size_t batch_size = (data_ptr_attr.devicePointer == nullptr) ? 100000 : nrow_;

  cuvs::spatial::knn::detail::utils::batch_load_iterator vec_batches{
    data, static_cast<size_t>(nrow_), build_config_.dataset_dim, batch_size, stream};
  for (auto const& batch : vec_batches) {
    if (d_data_float_.has_value()) {
      preprocess_data_kernel<<<
        batch.size(),
        raft::warp_size(),
        sizeof(DataT) * ceildiv(build_config_.dataset_dim, static_cast<size_t>(raft::warp_size())) *
          raft::warp_size(),
        stream>>>(batch.data(),
                  d_data_float_.value().data_handle(),
                  build_config_.dataset_dim,
                  l2_norms_.data_handle(),
                  batch.offset(),
                  build_config_.metric);
    } else {
      preprocess_data_kernel<<<
        batch.size(),
        raft::warp_size(),
        sizeof(DataT) * ceildiv(build_config_.dataset_dim, static_cast<size_t>(raft::warp_size())) *
          raft::warp_size(),
        stream>>>(batch.data(),
                  d_data_half_.value().data_handle(),
                  build_config_.dataset_dim,
                  l2_norms_.data_handle(),
                  batch.offset(),
                  build_config_.metric);
    }
  }

  graph_.clear();
  graph_.init_random_graph();
  graph_.sample_graph(true);

  auto update_and_sample = [&](bool update_graph) -> void {
    if (update_graph) {
      update_counter_ = 0;
      graph_.update_graph(graph_host_buffer_.data_handle(),
                          dists_host_buffer_.data_handle(),
                          kDegreeOnDevice,
                          update_counter_);
      if (update_counter_ < build_config_.termination_threshold * nrow_ *
                              build_config_.dataset_dim / kCounterInterval) {
        update_counter_ = -1;
      }
    }
    graph_.sample_graph(false);
  };

  for (size_t it = 0; it < build_config_.max_iterations; it++) {
    raft::copy(d_list_sizes_new_.data_handle(),
               graph_.h_list_sizes_new.data_handle(),
               nrow_,
               raft::resource::get_cuda_stream(res_));
    raft::copy(h_graph_old_.data_handle(),
               graph_.h_graph_old.data_handle(),
               nrow_ * kNumSamples,
               raft::resource::get_cuda_stream(res_));
    raft::copy(d_list_sizes_old_.data_handle(),
               graph_.h_list_sizes_old.data_handle(),
               nrow_,
               raft::resource::get_cuda_stream(res_));
    raft::resource::sync_stream(res_);

    std::thread update_and_sample_thread(update_and_sample, it);

    RAFT_LOG_DEBUG("# gnnd iteraton: %lu / %lu", it + 1, build_config_.max_iterations);

    // Reuse dists_buffer_ to save GPU memory. graph_buffer_ cannot be reused, because it
    // contains some information for local_join.
    static_assert(kDegreeOnDevice * sizeof(*(dists_buffer_.data_handle())) >=
                  kNumSamples * sizeof(*(graph_buffer_.data_handle())));
    add_reverse_edges(graph_.h_graph_new.data_handle(),
                      h_rev_graph_new_.data_handle(),
                      (index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_new_.data_handle(),
                      stream);
    add_reverse_edges(h_graph_old_.data_handle(),
                      h_rev_graph_old_.data_handle(),
                      (index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_old_.data_handle(),
                      stream);

    // Tensor operations from `mma.h` are guarded with archicteture
    // __CUDA_ARCH__ >= 700. Since RAFT supports compilation for ARCH 600,
    // we need to ensure that `local_join_kernel` (which uses tensor) operations
    // is not only not compiled, but also a runtime error is presented to the user
    auto kernel       = preprocess_data_kernel<input_t>;
    void* kernel_ptr  = reinterpret_cast<void*>(kernel);
    auto runtime_arch = raft::util::arch::kernel_virtual_arch(kernel_ptr);
    auto wmma_range =
      raft::util::arch::SM_range(raft::util::arch::SM_70(), raft::util::arch::SM_future());

    if (wmma_range.contains(runtime_arch)) {
      local_join(stream, dist_epilogue);
    } else {
      THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700");
    }

    update_and_sample_thread.join();

    if (update_counter_ == -1) { break; }
    raft::copy(graph_host_buffer_.data_handle(),
               graph_buffer_.data_handle(),
               nrow_ * kDegreeOnDevice,
               raft::resource::get_cuda_stream(res_));
    raft::copy(dists_host_buffer_.data_handle(),
               dists_buffer_.data_handle(),
               nrow_ * kDegreeOnDevice,
               raft::resource::get_cuda_stream(res_));
    raft::resource::sync_stream(res_);

    graph_.sample_graph_new(graph_host_buffer_.data_handle(), kDegreeOnDevice);
  }

  graph_.update_graph(graph_host_buffer_.data_handle(),
                      dists_host_buffer_.data_handle(),
                      kDegreeOnDevice,
                      update_counter_);
  raft::resource::sync_stream(res_);
  graph_.sort_lists();

  // Reuse graph_.h_dists as the buffer for shrink the lists in graph
  static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(index_t));

  if (return_distances) {
    auto graph_h_dists = raft::make_host_matrix<dist_data_t, int64_t, raft::row_major>(
      nrow_, build_config_.output_graph_degree);

// slice on host
#pragma omp parallel for
    for (size_t i = 0; i < (size_t)nrow_; i++) {
      for (size_t j = 0; j < build_config_.output_graph_degree; j++) {
        graph_h_dists(i, j) = graph_.h_dists(i, j);
      }
    }
    raft::copy(output_distances,
               graph_h_dists.data_handle(),
               nrow_ * build_config_.output_graph_degree,
               raft::resource::get_cuda_stream(res_));

    auto output_dist_view = raft::make_device_matrix_view<dist_data_t, int64_t, raft::row_major>(
      output_distances, nrow_, build_config_.output_graph_degree);
    // distance post-processing
    bool can_postprocess_dist = std::is_same_v<DistEpilogueT, raft::identity_op>;
    if (build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded &&
        can_postprocess_dist) {
      raft::linalg::map(
        res_, output_dist_view, raft::sqrt_op{}, raft::make_const_mdspan(output_dist_view));
    } else if (!cuvs::distance::is_min_close(build_config_.metric) && can_postprocess_dist) {
      // revert negated innerproduct
      raft::linalg::map(res_,
                        output_dist_view,
                        raft::mul_const_op<dist_data_t>(-1),
                        raft::make_const_mdspan(output_dist_view));
    }
    raft::resource::sync_stream(res_);
  }

  auto* graph_shrink_buffer = reinterpret_cast<index_t*>(graph_.h_dists.data_handle());

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      size_t idx = i * graph_.node_degree + j;
      int id     = graph_.h_graph[idx].id();
      if (id < static_cast<int>(nrow_)) {
        graph_shrink_buffer[i * build_config_.node_degree + j] = id;
      } else {
        graph_shrink_buffer[i * build_config_.node_degree + j] =
          cuvs::neighbors::cagra::detail::device::xorshift64(idx) % nrow_;
      }
    }
  }
  graph_.h_graph = nullptr;

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      output_graph[i * build_config_.node_degree + j] =
        graph_shrink_buffer[i * build_config_.node_degree + j];
    }
  }
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void build(raft::resources const& res,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
           index<IdxT>& idx)
{
  size_t extended_graph_degree, graph_degree;
  auto build_config = get_build_config(res,
                                       params,
                                       static_cast<size_t>(dataset.extent(0)),
                                       static_cast<size_t>(dataset.extent(1)),
                                       idx.metric(),
                                       extended_graph_degree,
                                       graph_degree);

  auto int_graph =
    raft::make_host_matrix<int, int64_t, raft::row_major>(dataset.extent(0), extended_graph_degree);

  gnnd<const T, int> nnd(res, build_config);

  if (idx.distances().has_value() || !params.return_distances) {
    nnd.build(dataset.data_handle(),
              dataset.extent(0),
              int_graph.data_handle(),
              params.return_distances,
              idx.distances()
                .value_or(raft::make_device_matrix<float, int64_t>(res, 0, 0).view())
                .data_handle());
  } else {
    RAFT_EXPECTS(!params.return_distances,
                 "Distance view not allocated. Using return_distances set to true requires "
                 "distance view to be allocated.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(dataset.extent(0)); i++) {
    for (size_t j = 0; j < graph_degree; j++) {
      auto graph                  = idx.graph().data_handle();
      graph[i * graph_degree + j] = int_graph.data_handle()[i * extended_graph_degree + j];
    }
  }
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
auto build(raft::resources const& res,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
  -> index<IdxT>
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  index<IdxT> idx{res,
                  dataset.extent(0),
                  static_cast<int64_t>(graph_degree),
                  params.return_distances,
                  params.metric};

  build(res, params, dataset, idx);

  return idx;
}

}  // namespace cuvs::neighbors::nn_descent::detail
