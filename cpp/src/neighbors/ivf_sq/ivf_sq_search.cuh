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

#include <rmm/resource_ref.hpp>

#include <thrust/fill.h>

namespace cuvs::neighbors::ivf_sq::detail {

using namespace cuvs::spatial::knn::detail;  // NOLINT

enum class SqScanMetric { kL2, kIP, kCosine };

/**
 * Per-probe scan kernel for IVF-SQ search.
 *
 * Grid: (n_queries, n_probes).  Each block handles one (query, probe) pair.
 * Within a block, each warp processes one interleaved group of kIndexGroupSize
 * (=32) vectors at a time, with each lane responsible for one vector.
 * Dimension blocks of veclen=16 bytes are loaded as coalesced uint4 reads
 * across the warp (32 lanes x 16 bytes = 512 bytes = 4 cache lines), giving
 * full memory-bandwidth utilisation.
 *
 * Per-dimension constants that are invariant across rows are precomputed into
 * shared memory so the hot loop only reads from smem + one uint4 per dim-block:
 *
 *   L2 / L2Sqrt:
 *     s_query_term[d] = query[d] - centroid[d] - sq_vmin[d]
 *     dist += (s_query_term[d] - code * s_sq_scale[d])^2
 *
 *   InnerProduct / Cosine:
 *     s_query_term[d] = query[d]
 *     s_recon_base[d] = centroid[d] + sq_vmin[d]
 *     v_d   = s_recon_base[d] + code * s_sq_scale[d]
 *     dist += s_query_term[d] * v_d
 *
 * Shared-memory layout adapts to the metric to avoid waste:
 *   L2 / L2Sqrt       : [s_query_term | s_sq_scale]                (2 * dim floats)
 *   InnerProduct/Cosine: [s_query_term | s_recon_base | s_sq_scale] (3 * dim floats)
 */
template <int BlockDim, SqScanMetric Metric, typename IdxT, typename IvfSampleFilterT>
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
                                                           uint32_t max_samples,
                                                           const uint32_t* chunk_indices,
                                                           float* out_distances,
                                                           uint32_t* out_indices,
                                                           IvfSampleFilterT sample_filter)
{
  static_assert(kIndexGroupSize == raft::WarpSize,
                "Warp-coalesced scan requires kIndexGroupSize == WarpSize");

  extern __shared__ float smem[];

  constexpr bool kIsL2     = (Metric == SqScanMetric::kL2);
  constexpr bool kIsCosine = (Metric == SqScanMetric::kCosine);

  float* s_query_term = smem;
  float* s_recon_base = smem + dim;
  float* s_sq_scale   = kIsL2 ? (smem + dim) : (smem + 2 * dim);

  const uint32_t query_ix = blockIdx.x;
  const uint32_t probe_ix = blockIdx.y;

  const uint32_t* my_coarse = coarse_indices + query_ix * n_probes;
  const uint32_t cluster_id = my_coarse[probe_ix];
  const uint32_t cluster_sz = list_sizes[cluster_id];
  if (cluster_sz == 0) return;

  const uint8_t* codes  = data_ptrs[cluster_id];
  const float* query    = queries_float + query_ix * dim;
  const float* centroid = centers + cluster_id * dim;

  for (uint32_t d = threadIdx.x; d < dim; d += BlockDim) {
    float vmin_d  = sq_vmin[d];
    s_sq_scale[d] = sq_delta[d];
    if constexpr (kIsL2) {
      s_query_term[d] = query[d] - centroid[d] - vmin_d;
    } else {
      s_query_term[d] = query[d];
      s_recon_base[d] = centroid[d] + vmin_d;
    }
  }
  __syncthreads();

  const uint32_t* my_chunk = chunk_indices + query_ix * n_probes;
  uint32_t out_base        = (probe_ix > 0) ? my_chunk[probe_ix - 1] : 0;

  constexpr uint32_t veclen         = 16;
  constexpr uint32_t kWarpsPerBlock = BlockDim / raft::WarpSize;
  const uint32_t warp_id            = threadIdx.x / raft::WarpSize;
  const uint32_t lane_id            = threadIdx.x % raft::WarpSize;

  uint32_t padded_dim   = ((dim + veclen - 1) / veclen) * veclen;
  uint32_t n_dim_blocks = padded_dim / veclen;

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
            float v_d = s_recon_base[l + j] + recon;
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

    if (valid) {
      uint32_t out_idx       = query_ix * max_samples + out_base + row;
      out_distances[out_idx] = dist;
      out_indices[out_idx]   = out_base + row;
    }
  }
}

template <typename IdxT, typename IvfSampleFilterT>
void ivf_sq_scan(raft::resources const& handle,
                 const index<IdxT>& idx,
                 const float* queries_float,
                 const float* query_norms,
                 uint32_t n_queries,
                 uint32_t n_probes,
                 uint32_t max_samples,
                 const uint32_t* coarse_indices,
                 const uint32_t* chunk_indices,
                 float* out_distances,
                 uint32_t* out_indices,
                 IvfSampleFilterT sample_filter,
                 rmm::cuda_stream_view stream)
{
  constexpr int kThreads = 256;
  dim3 grid(n_queries, n_probes);
  dim3 block(kThreads);
  uint32_t dim = idx.dim();

  auto do_launch = [&](auto kernel_ptr, size_t smem) {
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_ptr<<<grid, block, smem, stream>>>(idx.data_ptrs().data_handle(),
                                              idx.list_sizes().data_handle(),
                                              coarse_indices,
                                              queries_float,
                                              idx.centers().data_handle(),
                                              idx.sq_vmin().data_handle(),
                                              idx.sq_delta().data_handle(),
                                              query_norms,
                                              n_probes,
                                              dim,
                                              max_samples,
                                              chunk_indices,
                                              out_distances,
                                              out_indices,
                                              sample_filter);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };

  switch (idx.metric()) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2SqrtExpanded:
      do_launch(ivf_sq_scan_kernel<kThreads, SqScanMetric::kL2, IdxT, IvfSampleFilterT>,
                2 * dim * sizeof(float));
      break;
    case cuvs::distance::DistanceType::InnerProduct:
      do_launch(ivf_sq_scan_kernel<kThreads, SqScanMetric::kIP, IdxT, IvfSampleFilterT>,
                3 * dim * sizeof(float));
      break;
    case cuvs::distance::DistanceType::CosineExpanded:
      do_launch(ivf_sq_scan_kernel<kThreads, SqScanMetric::kCosine, IdxT, IvfSampleFilterT>,
                3 * dim * sizeof(float));
      break;
    default: RAFT_FAIL("Unsupported metric type for IVF-SQ scan.");
  }
}

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

  rmm::device_uvector<float> query_norm_dev(n_queries, stream, search_mr);
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
      [=] __device__(const uint32_t idx, const float dist) {
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

  uint32_t max_samples =
    std::max<uint32_t>(static_cast<uint32_t>(index.accum_sorted_sizes()(n_probes)), k);

  rmm::device_uvector<float> all_distances(std::size_t(n_queries) * max_samples, stream, search_mr);
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

  auto filter_adapter = cuvs::neighbors::filtering::ivf_to_sample_filter(
    index.inds_ptrs().data_handle(), sample_filter);

  ivf_sq_scan(handle,
              index,
              converted_queries_ptr,
              query_norm_dev.data(),
              n_queries,
              n_probes,
              max_samples,
              coarse_indices_dev.data(),
              chunk_index.data(),
              all_distances.data(),
              all_indices.data(),
              filter_adapter,
              stream);

  rmm::device_uvector<uint32_t> neighbors_uint32(0, stream, search_mr);
  uint32_t* neighbors_uint32_ptr = nullptr;
  if constexpr (sizeof(int64_t) == sizeof(uint32_t)) {
    neighbors_uint32_ptr = reinterpret_cast<uint32_t*>(neighbors);
  } else {
    neighbors_uint32.resize(std::size_t(n_queries) * k, stream);
    neighbors_uint32_ptr = neighbors_uint32.data();
  }

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

  ivf::detail::postprocess_distances(
    distances, distances, index.metric(), n_queries, k, 1.0, false, stream);

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

  uint32_t max_samples =
    std::max<uint32_t>(static_cast<uint32_t>(index.accum_sorted_sizes()(n_probes)), k);

  constexpr uint64_t kExpectedWsSize = 1024ull * 1024 * 1024;
  uint64_t max_ws_size =
    std::min<uint64_t>(raft::resource::get_workspace_free_bytes(handle), kExpectedWsSize);

  uint64_t converted_query_floats = std::is_same_v<T, float> ? 0 : index.dim();
  uint64_t ws_per_query = sizeof(float) * (uint64_t(index.n_lists()) + n_probes + 1 + max_samples +
                                           converted_query_floats) +
                          sizeof(uint32_t) * (uint64_t(n_probes) * 2 + 1 + max_samples + k);

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
