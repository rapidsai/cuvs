/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../core/nvtx.hpp"
#include "../detail/ann_utils.cuh"
#include "../ivf_common.cuh"
#include "../sample_filter.cuh"
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/selection/select_k.hpp>
#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/util/integer_utils.hpp>

#include <rmm/resource_ref.hpp>

#include <thrust/fill.h>

namespace cuvs::neighbors::ivf_sq::detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

enum class SqScanMetric { kL2, kIP, kCosine };

static constexpr int kSqScanThreads = 128;

// Maximum fused top-k capacity we instantiate for the scan kernel.
// Must match the highest Capacity case in ivf_sq_scan's switch.
static constexpr int kMaxSqScanCapacity = 256;
static_assert(kMaxSqScanCapacity <= raft::matrix::detail::select::warpsort::kMaxCapacity,
              "kMaxSqScanCapacity must not exceed the warpsort library's maximum supported "
              "capacity; reduce kMaxSqScanCapacity or update the warpsort dependency.");

auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k) -> bool
{
  return k <= kMaxSqScanCapacity;
}

// ---------------------------------------------------------------------------
// block_sort type selection (fused top-k vs dummy for Capacity == 0)
// ---------------------------------------------------------------------------
template <int Capacity, bool Ascending>
struct sq_block_sort {
  using type = raft::matrix::detail::select::warpsort::block_sort<
    raft::matrix::detail::select::warpsort::warp_sort_filtered,
    Capacity,
    Ascending,
    float,
    uint32_t>;
};

template <bool Ascending>
struct sq_block_sort<0, Ascending> {
  using type = ivf::detail::dummy_block_sort_t<float, uint32_t, Ascending>;
};

template <int Capacity, bool Ascending>
using sq_block_sort_t = typename sq_block_sort<Capacity, Ascending>::type;

// ---------------------------------------------------------------------------
// configure_grid_dim_x: choose grid.x to saturate the GPU
// ---------------------------------------------------------------------------
inline uint32_t configure_grid_dim_x(
  uint32_t n_queries, uint32_t n_probes, int smem_size, int block_size, const void* kernel_ptr)
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_sms;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, kernel_ptr, block_size, smem_size));

  size_t min_grid_size = size_t(num_sms) * num_blocks_per_sm;
  size_t min_grid_x    = raft::ceildiv<size_t>(min_grid_size, n_queries);
  return std::min<uint32_t>(n_probes, static_cast<uint32_t>(min_grid_x));
}

// ---------------------------------------------------------------------------
// IVF-SQ scan kernel with fused in-kernel top-k
//
// Grid layout:
//   kManageLocalTopK (Capacity > 0):
//     grid (grid_dim_x, n_queries) — each block loops over probes
//   otherwise (Capacity == 0):
//     grid (n_probes, n_queries) — one block per (query, probe)
//
// Shared-memory layout (always 3 × dim floats):
//   [s_sq_scale(dim) | s_query_term(dim) | s_aux(dim)]
//
//   s_sq_scale = delta[d]  — SQ dequantization scale, invariant (Phase 1).
//
//   L2 path:
//     Phase 1: s_aux[d] = query[d] - vmin[d]               (invariant)
//     Phase 2: s_query_term[d] = s_aux[d] - centroid[d]     (per-probe)
//     The full SQ reconstruction is centroid + vmin + code*delta, so
//     query - reconstructed = (query - vmin - centroid) - code*delta
//                           = s_query_term - code*s_sq_scale.
//
//   IP/Cosine path:
//     Phase 1: s_query_term[d] = query[d]                   (invariant)
//     Phase 2: s_aux[d] = centroid[d] + vmin[d]             (per-probe)
//     Reconstructed vector component: s_aux[d] + code*s_sq_scale[d].
//
//   After all probes are scanned, the smem is reused for block_sort merge.
// ---------------------------------------------------------------------------
template <int BlockDim, int Capacity, SqScanMetric Metric, typename IdxT, typename IvfSampleFilterT>
__launch_bounds__(BlockDim) RAFT_KERNEL ivf_sq_scan_kernel(const uint8_t* const* data_ptrs,
                                                           const uint32_t* list_sizes,
                                                           const uint32_t* coarse_indices,
                                                           const float* queries_float,
                                                           const float* centers,
                                                           const float* sq_vmin,
                                                           const float* sq_delta,
                                                           const float* query_norms,
                                                           uint32_t n_probes,
                                                           uint32_t dim,
                                                           uint32_t k,
                                                           uint32_t max_samples,
                                                           const uint32_t* chunk_indices,
                                                           float* out_distances,
                                                           uint32_t* out_indices,
                                                           IvfSampleFilterT sample_filter)
{
  static_assert(kIndexGroupSize == raft::WarpSize,
                "Warp-coalesced scan requires kIndexGroupSize == WarpSize");

  constexpr bool kManageLocalTopK = (Capacity > 0);
  constexpr bool kIsL2            = (Metric == SqScanMetric::kL2);
  constexpr bool kIsCosine        = (Metric == SqScanMetric::kCosine);
  constexpr bool kAscending       = (Metric != SqScanMetric::kIP);

  extern __shared__ __align__(256) uint8_t smem_buf[];
  float* smem = reinterpret_cast<float*>(smem_buf);

  float* s_sq_scale   = smem;
  float* s_query_term = smem + dim;
  float* s_aux        = smem + 2 * dim;

  const uint32_t query_ix = blockIdx.y;
  const float* query      = queries_float + query_ix * dim;

  // Point output to this block's slice when using fused top-k
  if constexpr (kManageLocalTopK) {
    out_distances += uint64_t(query_ix) * k * gridDim.x + blockIdx.x * k;
    out_indices += uint64_t(query_ix) * k * gridDim.x + blockIdx.x * k;
  }

  // --- Phase 1: load shared memory that is invariant across probes ---
  for (uint32_t d = threadIdx.x; d < dim; d += BlockDim) {
    s_sq_scale[d] = sq_delta[d];
    if constexpr (kIsL2) {
      s_aux[d] = query[d] - sq_vmin[d];
    } else {
      s_query_term[d] = query[d];
    }
  }
  __syncthreads();

  using local_topk_t = sq_block_sort_t<Capacity, kAscending>;
  local_topk_t queue(k);

  const uint32_t* my_coarse = coarse_indices + query_ix * n_probes;
  const uint32_t* my_chunk  = chunk_indices + query_ix * n_probes;

  constexpr uint32_t veclen         = 16;
  constexpr uint32_t kWarpsPerBlock = BlockDim / raft::WarpSize;
  const uint32_t warp_id            = threadIdx.x / raft::WarpSize;
  const uint32_t lane_id            = threadIdx.x % raft::WarpSize;

  // --- Phase 2: loop over probes ---
  // Synchronization protocol:
  //  (a) __syncthreads after Phase 1 (above) ensures invariant smem arrays
  //      (s_sq_scale, and L2: s_aux / IP-Cosine: s_query_term) are visible
  //      before Phase 2 overwrites the per-probe array.
  //  (b) __syncthreads after per-probe smem writes (L2: s_query_term /
  //      IP-Cosine: s_aux) ensures probe-specific values are visible before
  //      the distance computation.
  //  (c) __syncthreads at the end of each iteration ensures all distance
  //      computation reads are complete before the next iteration overwrites
  //      the per-probe smem region.
  //  When cluster_sz == 0, barrier (c) is skipped because no distance reads
  //  occurred; all threads converge on the same branch uniformly, and the
  //  next iteration's barrier (b) provides the needed ordering.
  for (uint32_t probe_ix = blockIdx.x; probe_ix < n_probes;
       probe_ix += (kManageLocalTopK ? gridDim.x : uint32_t{1})) {
    const uint32_t cluster_id = my_coarse[probe_ix];
    const uint32_t cluster_sz = list_sizes[cluster_id];

    // Load centroid-dependent shared memory terms
    {
      const float* centroid = centers + cluster_id * dim;
      for (uint32_t d = threadIdx.x; d < dim; d += BlockDim) {
        if constexpr (kIsL2) {
          s_query_term[d] = s_aux[d] - centroid[d];
        } else {
          s_aux[d] = centroid[d] + sq_vmin[d];
        }
      }
    }
    __syncthreads();  // (b)

    if (cluster_sz == 0) {
      // No distance computation reads happened, so no end-of-iteration
      // barrier is needed; the next iteration's barrier (b) is sufficient.
      if constexpr (!kManageLocalTopK) break;
      continue;
    }

    const uint8_t* codes   = data_ptrs[cluster_id];
    uint32_t sample_offset = (probe_ix > 0) ? my_chunk[probe_ix - 1] : 0;
    uint32_t padded_dim    = ((dim + veclen - 1) / veclen) * veclen;
    uint32_t n_dim_blocks  = padded_dim / veclen;

    for (uint32_t group = warp_id * kIndexGroupSize; group < cluster_sz;
         group += kWarpsPerBlock * kIndexGroupSize) {
      const uint32_t row = group + lane_id;
      const bool valid   = (row < cluster_sz) && sample_filter(query_ix, cluster_id, row);

      float dist      = 0.0f;
      float v_norm_sq = 0.0f;

      const uint8_t* group_data = codes + size_t(group) * padded_dim;

      for (uint32_t bl = 0; bl < n_dim_blocks; bl++) {
        uint8_t codes_local[veclen];
        *reinterpret_cast<uint4*>(codes_local) = *reinterpret_cast<const uint4*>(
          group_data + bl * (veclen * kIndexGroupSize) + lane_id * veclen);

        const uint32_t l = bl * veclen;
#pragma unroll
        for (uint32_t j = 0; j < veclen; j++) {
          if (l + j < dim) {
            float recon = float(codes_local[j]) * s_sq_scale[l + j];

            if constexpr (kIsL2) {
              float diff = s_query_term[l + j] - recon;
              dist += diff * diff;
            } else {
              float v_d = s_aux[l + j] + recon;
              dist += s_query_term[l + j] * v_d;
              if constexpr (kIsCosine) { v_norm_sq += v_d * v_d; }
            }
          }
        }
      }

      if constexpr (kIsCosine) {
        float denom = query_norms[query_ix] * sqrtf(v_norm_sq);
        dist        = (denom > 0.0f) ? 1.0f - dist / denom : 0.0f;
      }

      if constexpr (kManageLocalTopK) {
        float val = valid ? dist : local_topk_t::queue_t::kDummy;
        queue.add(val, sample_offset + row);
      } else {
        if (valid) {
          uint32_t out_idx       = query_ix * max_samples + sample_offset + row;
          out_distances[out_idx] = dist;
          out_indices[out_idx]   = sample_offset + row;
        }
      }
    }

    __syncthreads();  // (c)
    if constexpr (!kManageLocalTopK) break;
  }

  if constexpr (kManageLocalTopK) {
    // All probe iterations are done; smem_buf is reused for block_sort merge.
    // The loop's last (b) or (c) barrier ensures all prior smem accesses have
    // completed, so this additional barrier is only needed to synchronize any
    // register-level state across warps before the merge.
    __syncthreads();
    queue.done(smem_buf);
    queue.store(out_distances, out_indices);

    // block_sort initializes unused slots with (kDummy, idx=0). When the
    // probed clusters have fewer than k total valid vectors, those slots
    // survive into the output and share idx=0 with the real first vector,
    // causing duplicates.  Mark them with an invalid index so
    // postprocess_neighbors treats them as out-of-bounds.
    // store() is a warp-0-only operation, restrict the fixup to the same warp.
    if (threadIdx.x < raft::WarpSize) {
      constexpr auto kDummyVal = local_topk_t::queue_t::kDummy;
      for (uint32_t i = threadIdx.x; i < k; i += raft::WarpSize) {
        if (out_distances[i] == kDummyVal) { out_indices[i] = uint32_t(0xFFFFFFFF); }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Compute shared-memory size for a given kernel configuration
// ---------------------------------------------------------------------------
inline size_t sq_scan_smem_size(uint32_t dim) { return 3 * dim * sizeof(float); }

template <int Capacity>
size_t sq_scan_total_smem(uint32_t dim, uint32_t k)
{
  size_t scan_smem = sq_scan_smem_size(dim);
  if constexpr (Capacity > 0) {
    constexpr int kSubwarpSize = std::min<int>(Capacity, raft::WarpSize);
    int num_subwarps           = kSqScanThreads / kSubwarpSize;
    size_t merge_smem =
      raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<float, uint32_t>(
        num_subwarps, k);
    return std::max(scan_smem, merge_smem);
  }
  return scan_smem;
}

// ---------------------------------------------------------------------------
// Launch helper: dispatches on Metric, handles grid_dim_x query vs launch
// ---------------------------------------------------------------------------
template <int Capacity, typename IdxT, typename IvfSampleFilterT>
void ivf_sq_scan_launch(const index<IdxT>& idx,
                        const float* queries_float,
                        const float* query_norms,
                        uint32_t n_queries,
                        uint32_t n_probes,
                        uint32_t k,
                        uint32_t max_samples,
                        const uint32_t* coarse_indices,
                        const uint32_t* chunk_indices,
                        float* out_distances,
                        uint32_t* out_indices,
                        IvfSampleFilterT sample_filter,
                        uint32_t& grid_dim_x,
                        rmm::cuda_stream_view stream)
{
  constexpr bool kManageLocalTopK = (Capacity > 0);
  constexpr int kThreads          = kSqScanThreads;
  uint32_t dim                    = idx.dim();

  constexpr uint32_t kMaxGridY = 65535;

  auto do_launch = [&](auto kernel_ptr) {
    size_t smem = sq_scan_total_smem<Capacity>(dim, k);

    {
      int dev_id;
      RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
      int max_smem;
      RAFT_CUDA_TRY(
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_id));
      RAFT_EXPECTS(smem <= size_t(max_smem),
                   "IVF-SQ scan kernel requires %zu bytes of shared memory (dim=%u, k=%u), "
                   "but the device supports at most %d bytes per block.",
                   smem,
                   dim,
                   k,
                   max_smem);
    }

    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    // If grid_dim_x == 0, compute the optimal value and return
    if constexpr (kManageLocalTopK) {
      if (grid_dim_x == 0) {
        grid_dim_x = configure_grid_dim_x(std::min(kMaxGridY, n_queries),
                                          n_probes,
                                          smem,
                                          kThreads,
                                          reinterpret_cast<const void*>(kernel_ptr));
        return;
      }
    }

    dim3 block(kThreads);

    // Batch over queries to respect the gridDim.y limit (65535)
    for (uint32_t query_offset = 0; query_offset < n_queries; query_offset += kMaxGridY) {
      uint32_t batch = std::min(kMaxGridY, n_queries - query_offset);
      dim3 grid      = kManageLocalTopK ? dim3(grid_dim_x, batch) : dim3(n_probes, batch);

      auto q_ptr  = queries_float + uint64_t(query_offset) * dim;
      auto qn_ptr = query_norms ? query_norms + query_offset : query_norms;
      auto ci     = coarse_indices + uint64_t(query_offset) * n_probes;
      auto ch     = chunk_indices + uint64_t(query_offset) * n_probes;
      auto od     = out_distances;
      auto oi     = out_indices;
      if constexpr (kManageLocalTopK) {
        od += uint64_t(query_offset) * grid_dim_x * k;
        oi += uint64_t(query_offset) * grid_dim_x * k;
      } else {
        od += uint64_t(query_offset) * max_samples;
        oi += uint64_t(query_offset) * max_samples;
      }

      kernel_ptr<<<grid, block, smem, stream>>>(idx.data_ptrs().data_handle(),
                                                idx.list_sizes().data_handle(),
                                                ci,
                                                q_ptr,
                                                idx.centers().data_handle(),
                                                idx.sq_vmin().data_handle(),
                                                idx.sq_delta().data_handle(),
                                                qn_ptr,
                                                n_probes,
                                                dim,
                                                k,
                                                max_samples,
                                                ch,
                                                od,
                                                oi,
                                                sample_filter);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  };

  switch (idx.metric()) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
      do_launch(ivf_sq_scan_kernel<kThreads, Capacity, SqScanMetric::kL2, IdxT, IvfSampleFilterT>);
      break;
    case cuvs::distance::DistanceType::InnerProduct:
      do_launch(ivf_sq_scan_kernel<kThreads, Capacity, SqScanMetric::kIP, IdxT, IvfSampleFilterT>);
      break;
    case cuvs::distance::DistanceType::CosineExpanded:
      do_launch(
        ivf_sq_scan_kernel<kThreads, Capacity, SqScanMetric::kCosine, IdxT, IvfSampleFilterT>);
      break;
    default: RAFT_FAIL("Unsupported metric type for IVF-SQ scan.");
  }
}

// ---------------------------------------------------------------------------
// ivf_sq_scan: top-level scan dispatch with Capacity selection
// ---------------------------------------------------------------------------
template <typename IdxT, typename IvfSampleFilterT>
void ivf_sq_scan(raft::resources const& handle,
                 const index<IdxT>& idx,
                 const float* queries_float,
                 const float* query_norms,
                 uint32_t n_queries,
                 uint32_t n_probes,
                 uint32_t k,
                 uint32_t max_samples,
                 const uint32_t* coarse_indices,
                 const uint32_t* chunk_indices,
                 float* out_distances,
                 uint32_t* out_indices,
                 IvfSampleFilterT sample_filter,
                 uint32_t& grid_dim_x,
                 rmm::cuda_stream_view stream)
{
  // Determine the fused top-k capacity (0 = disabled / fallback to materialization)
  int capacity = is_local_topk_feasible(k) ? raft::bound_by_power_of_two(int(k)) : 0;

  // Snap to the nearest supported compile-time Capacity value (must be a
  // power of two). Values up to 32 share one instantiation; 64, 128 and 256
  // each get their own.  Beyond kMaxSqScanCapacity we fall back to the
  // non-fused path (Capacity == 0).
  if (capacity > 0 && capacity < 32) {
    capacity = 32;
  } else if (capacity > kMaxSqScanCapacity) {
    capacity = 0;
  }

  auto fwd = [&](auto cap_tag) {
    ivf_sq_scan_launch<decltype(cap_tag)::value, IdxT, IvfSampleFilterT>(idx,
                                                                         queries_float,
                                                                         query_norms,
                                                                         n_queries,
                                                                         n_probes,
                                                                         k,
                                                                         max_samples,
                                                                         coarse_indices,
                                                                         chunk_indices,
                                                                         out_distances,
                                                                         out_indices,
                                                                         sample_filter,
                                                                         grid_dim_x,
                                                                         stream);
  };

  switch (capacity) {
    case 0: fwd(std::integral_constant<int, 0>{}); break;
    case 32: fwd(std::integral_constant<int, 32>{}); break;
    case 64: fwd(std::integral_constant<int, 64>{}); break;
    case 128: fwd(std::integral_constant<int, 128>{}); break;
    case 256: fwd(std::integral_constant<int, 256>{}); break;
    default: RAFT_FAIL("Unexpected capacity value %d", capacity);
  }
}

// ---------------------------------------------------------------------------
// search_impl — host-side search logic
// ---------------------------------------------------------------------------
template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_impl(raft::resources const& handle,
                 const index<IdxT>& index,
                 const T* queries,
                 uint32_t n_queries,
                 uint32_t k,
                 uint32_t n_probes,
                 bool select_min,
                 int64_t* neighbors,
                 float* distances,
                 rmm::device_async_resource_ref search_mr,
                 IvfSampleFilterT sample_filter)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  auto dim    = index.dim();

  std::size_t n_queries_probes = std::size_t(n_queries) * std::size_t(n_probes);

  bool needs_query_norms = index.metric() != cuvs::distance::DistanceType::InnerProduct;
  rmm::device_uvector<float> query_norm_dev(needs_query_norms ? n_queries : 0, stream, search_mr);
  rmm::device_uvector<float> distance_buffer_dev(n_queries * index.n_lists(), stream, search_mr);
  rmm::device_uvector<float> coarse_distances_dev(n_queries_probes, stream, search_mr);
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries_probes, stream, search_mr);

  size_t float_query_size;
  if constexpr (std::is_same_v<T, float>) {
    float_query_size = 0;
  } else {
    float_query_size = n_queries * dim;
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    raft::linalg::unaryOp(
      converted_queries_ptr, queries, n_queries * dim, utils::mapping<float>{}, stream);
  }

  auto distance_buffer_dev_view = raft::make_device_matrix_view<float, int64_t>(
    distance_buffer_dev.data(), n_queries, index.n_lists());

  RAFT_EXPECTS(index.metric() == cuvs::distance::DistanceType::InnerProduct ||
                 index.center_norms().has_value(),
               "Center norms are required for search with L2 or Cosine metric. "
               "Rebuild the index with add_data_on_build=true or call extend() first.");

  float alpha = 1.0f;
  float beta  = 0.0f;
  switch (index.metric()) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded: {
      alpha = -2.0f;
      beta  = 1.0f;
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(query_norm_dev.data(),
                                                        converted_queries_ptr,
                                                        static_cast<int64_t>(dim),
                                                        static_cast<int64_t>(n_queries),
                                                        stream);
      utils::outer_add(query_norm_dev.data(),
                       (int64_t)n_queries,
                       index.center_norms()->data_handle(),
                       (int64_t)index.n_lists(),
                       distance_buffer_dev.data(),
                       stream);
      break;
    }
    case cuvs::distance::DistanceType::CosineExpanded: {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(query_norm_dev.data(),
                                                        converted_queries_ptr,
                                                        static_cast<int64_t>(dim),
                                                        static_cast<int64_t>(n_queries),
                                                        stream,
                                                        raft::sqrt_op{});
      alpha = -1.0f;
      beta  = 0.0f;
      break;
    }
    case cuvs::distance::DistanceType::InnerProduct: {
      alpha = 1.0f;
      beta  = 0.0f;
      break;
    }
    default: RAFT_FAIL("Unsupported metric type for IVF-SQ search.");
  }

  raft::linalg::gemm(handle,
                     true,
                     false,
                     index.n_lists(),
                     n_queries,
                     dim,
                     &alpha,
                     index.centers().data_handle(),
                     dim,
                     converted_queries_ptr,
                     dim,
                     &beta,
                     distance_buffer_dev.data(),
                     index.n_lists(),
                     stream);

  if (index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
    auto n_lists_local          = index.n_lists();
    const auto* q_norm_ptr      = query_norm_dev.data();
    const auto* center_norm_ptr = index.center_norms()->data_handle();
    raft::linalg::map_offset(
      handle,
      distance_buffer_dev_view,
      [=] __device__(const int64_t idx, const float dist) {
        const auto query   = idx / n_lists_local;
        const auto cluster = idx % n_lists_local;
        float denom        = q_norm_ptr[query] * center_norm_ptr[cluster];
        return (denom > 0.0f) ? dist / denom : 0.0f;
      },
      raft::make_const_mdspan(distance_buffer_dev_view));
  }

  cuvs::selection::select_k(
    handle,
    raft::make_const_mdspan(distance_buffer_dev_view),
    std::nullopt,
    raft::make_device_matrix_view<float, int64_t>(coarse_distances_dev.data(), n_queries, n_probes),
    raft::make_device_matrix_view<uint32_t, int64_t>(
      coarse_indices_dev.data(), n_queries, n_probes),
    select_min);

  rmm::device_uvector<uint32_t> num_samples(n_queries, stream, search_mr);
  rmm::device_uvector<uint32_t> chunk_index(n_queries_probes, stream, search_mr);

  ivf::detail::calc_chunk_indices::configure(n_probes, n_queries)(index.list_sizes().data_handle(),
                                                                  coarse_indices_dev.data(),
                                                                  chunk_index.data(),
                                                                  num_samples.data(),
                                                                  stream);

  auto filter_adapter = cuvs::neighbors::filtering::ivf_to_sample_filter(
    index.inds_ptrs().data_handle(), sample_filter);

  bool manage_local_topk = is_local_topk_feasible(k);

  // Determine grid_dim_x for the fused path
  uint32_t grid_dim_x = 0;
  if (manage_local_topk) {
    // Query the occupancy to compute optimal grid_dim_x (does not launch)
    ivf_sq_scan(handle,
                index,
                converted_queries_ptr,
                query_norm_dev.data(),
                n_queries,
                n_probes,
                k,
                0,
                coarse_indices_dev.data(),
                chunk_index.data(),
                nullptr,
                nullptr,
                filter_adapter,
                grid_dim_x,
                stream);
    if (grid_dim_x == 0) {
      manage_local_topk = false;
      RAFT_LOG_WARN(
        "IVF-SQ fused top-k kernel has zero occupancy (dim=%u, k=%u); "
        "falling back to the non-fused scan path.",
        index.dim(),
        k);
    }
  }

  // Prepare uint32 neighbors buffer for postprocessing
  rmm::device_uvector<uint32_t> neighbors_uint32(0, stream, search_mr);
  uint32_t* neighbors_uint32_ptr = nullptr;
  if constexpr (sizeof(int64_t) == sizeof(uint32_t)) {
    neighbors_uint32_ptr = reinterpret_cast<uint32_t*>(neighbors);
  } else {
    neighbors_uint32.resize(std::size_t(n_queries) * k, stream);
    neighbors_uint32_ptr = neighbors_uint32.data();
  }

  if (manage_local_topk) {
    // --- Fused top-k path ---
    auto target_size = std::size_t(n_queries) * grid_dim_x * k;
    rmm::device_uvector<float> distances_tmp(0, stream, search_mr);
    rmm::device_uvector<uint32_t> indices_tmp(0, stream, search_mr);

    float* dist_out_ptr   = nullptr;
    uint32_t* idx_out_ptr = nullptr;

    if (grid_dim_x > 1) {
      distances_tmp.resize(target_size, stream);
      indices_tmp.resize(target_size, stream);
      dist_out_ptr = distances_tmp.data();
      idx_out_ptr  = indices_tmp.data();
    } else {
      dist_out_ptr = distances;
      idx_out_ptr  = neighbors_uint32_ptr;
    }

    ivf_sq_scan(handle,
                index,
                converted_queries_ptr,
                query_norm_dev.data(),
                n_queries,
                n_probes,
                k,
                0,
                coarse_indices_dev.data(),
                chunk_index.data(),
                dist_out_ptr,
                idx_out_ptr,
                filter_adapter,
                grid_dim_x,
                stream);

    // Merge across blocks if needed
    if (grid_dim_x > 1) {
      auto cols = uint32_t(grid_dim_x) * k;
      cuvs::selection::select_k(
        handle,
        raft::make_device_matrix_view<const float, int64_t>(distances_tmp.data(), n_queries, cols),
        raft::make_device_matrix_view<const uint32_t, int64_t>(indices_tmp.data(), n_queries, cols),
        raft::make_device_matrix_view<float, int64_t>(distances, n_queries, k),
        raft::make_device_matrix_view<uint32_t, int64_t>(neighbors_uint32_ptr, n_queries, k),
        select_min);
    }
  } else {
    // --- Fallback: materialize all distances ---
    int64_t ms = std::max<int64_t>(index.accum_sorted_sizes()(n_probes), k);
    RAFT_EXPECTS(ms <= int64_t(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
    uint32_t max_samples = static_cast<uint32_t>(ms);

    rmm::device_uvector<float> all_distances(
      std::size_t(n_queries) * max_samples, stream, search_mr);
    rmm::device_uvector<uint32_t> all_indices(
      std::size_t(n_queries) * max_samples, stream, search_mr);

    float init_val =
      select_min ? std::numeric_limits<float>::max() : std::numeric_limits<float>::lowest();
    thrust::fill_n(raft::resource::get_thrust_policy(handle),
                   all_distances.data(),
                   std::size_t(n_queries) * max_samples,
                   init_val);
    thrust::fill_n(raft::resource::get_thrust_policy(handle),
                   all_indices.data(),
                   std::size_t(n_queries) * max_samples,
                   uint32_t(0xFFFFFFFF));

    // grid_dim_x is unused for the non-fused path; set to n_probes so each
    // block in the (n_probes, n_queries) grid processes exactly one probe
    uint32_t gdx = n_probes;
    ivf_sq_scan(handle,
                index,
                converted_queries_ptr,
                query_norm_dev.data(),
                n_queries,
                n_probes,
                k,
                max_samples,
                coarse_indices_dev.data(),
                chunk_index.data(),
                all_distances.data(),
                all_indices.data(),
                filter_adapter,
                gdx,
                stream);

    auto num_samples_view =
      raft::make_device_vector_view<const uint32_t>(num_samples.data(), n_queries);

    cuvs::selection::select_k(
      handle,
      raft::make_device_matrix_view<const float, int64_t>(
        all_distances.data(), n_queries, max_samples),
      raft::make_device_matrix_view<const uint32_t, int64_t>(
        all_indices.data(), n_queries, max_samples),
      raft::make_device_matrix_view<float, int64_t>(distances, n_queries, k),
      raft::make_device_matrix_view<uint32_t, int64_t>(neighbors_uint32_ptr, n_queries, k),
      select_min,
      false,
      cuvs::selection::SelectAlgo::kAuto,
      num_samples_view);
  }

  ivf::detail::postprocess_distances(
    handle, distances, distances, index.metric(), n_queries, k, 1.0, false);

  ivf::detail::postprocess_neighbors(neighbors,
                                     neighbors_uint32_ptr,
                                     index.inds_ptrs().data_handle(),
                                     coarse_indices_dev.data(),
                                     chunk_index.data(),
                                     n_queries,
                                     n_probes,
                                     k,
                                     stream);
}

template <typename T,
          typename IdxT,
          typename IvfSampleFilterT = cuvs::neighbors::filtering::none_sample_filter>
inline void search_with_filtering(raft::resources const& handle,
                                  const search_params& params,
                                  const index<IdxT>& index,
                                  const T* queries,
                                  uint32_t n_queries,
                                  uint32_t k,
                                  int64_t* neighbors,
                                  float* distances,
                                  IvfSampleFilterT sample_filter = IvfSampleFilterT())
{
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "ivf_sq::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());
  if (n_probes < params.n_probes) {
    RAFT_LOG_WARN(
      "n_probes (%u) is larger than the number of lists in the index (%u), clamping to %u.",
      params.n_probes,
      index.n_lists(),
      n_probes);
  }

  bool manage_local_topk = is_local_topk_feasible(k);

  uint32_t max_samples = 0;
  if (!manage_local_topk) {
    int64_t ms = std::max<int64_t>(index.accum_sorted_sizes()(n_probes), k);
    RAFT_EXPECTS(ms <= int64_t(std::numeric_limits<uint32_t>::max()),
                 "The maximum sample size is too big.");
    max_samples = static_cast<uint32_t>(ms);
  }

  constexpr uint64_t kExpectedWsSize = 1024ull * 1024 * 1024;
  uint64_t max_ws_size =
    std::min<uint64_t>(raft::resource::get_workspace_free_bytes(handle), kExpectedWsSize);

  uint64_t converted_query_floats = std::is_same_v<T, float> ? 0 : index.dim();
  uint64_t ws_per_query;
  if (manage_local_topk) {
    // Fused path: only small per-query buffers for coarse search + chunk indices
    // (The scan output is at most grid_dim_x * k per query, which is small)
    // Conservatively assume grid_dim_x <= n_probes for the workspace estimate
    uint64_t fused_out = uint64_t(n_probes) * k;
    ws_per_query       = sizeof(float) * (uint64_t(index.n_lists()) + n_probes + 1 + fused_out +
                                    converted_query_floats) +
                   sizeof(uint32_t) * (uint64_t(n_probes) * 2 + 1 + fused_out + k);
  } else {
    ws_per_query = sizeof(float) * (uint64_t(index.n_lists()) + n_probes + 1 + max_samples +
                                    converted_query_floats) +
                   sizeof(uint32_t) * (uint64_t(n_probes) * 2 + 1 + max_samples + k);
  }

  const uint32_t max_queries =
    std::min<uint32_t>(n_queries, std::max<uint64_t>(1, max_ws_size / ws_per_query));

  for (uint32_t offset_q = 0; offset_q < n_queries; offset_q += max_queries) {
    uint32_t queries_batch = std::min(max_queries, n_queries - offset_q);

    search_impl<T, IdxT, IvfSampleFilterT>(handle,
                                           index,
                                           queries + std::size_t(offset_q) * index.dim(),
                                           queries_batch,
                                           k,
                                           n_probes,
                                           cuvs::distance::is_min_close(index.metric()),
                                           neighbors + std::size_t(offset_q) * k,
                                           distances + std::size_t(offset_q) * k,
                                           raft::resource::get_workspace_resource(handle),
                                           sample_filter);
  }
}

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<IdxT>& index,
                           raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                           raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           IvfSampleFilterT sample_filter = IvfSampleFilterT())
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");
  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must be equal");
  RAFT_EXPECTS(queries.extent(1) == index.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  search_with_filtering(handle,
                        params,
                        index,
                        queries.data_handle(),
                        static_cast<std::uint32_t>(queries.extent(0)),
                        static_cast<std::uint32_t>(neighbors.extent(1)),
                        neighbors.data_handle(),
                        distances.data_handle(),
                        sample_filter);
}

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<IdxT>& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter_ref)
{
  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::none_sample_filter&>(sample_filter_ref);
    return search_with_filtering(handle, params, idx, queries, neighbors, distances, sample_filter);
  } catch (const std::bad_cast&) {
  }

  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
        sample_filter_ref);
    return search_with_filtering(handle, params, idx, queries, neighbors, distances, sample_filter);
  } catch (const std::bad_cast&) {
    RAFT_FAIL("Unsupported sample filter type");
  }
}

}  // namespace cuvs::neighbors::ivf_sq::detail
