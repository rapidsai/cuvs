/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../detail/jit_lto_kernels/filter_data.h"
#include "../ivf_common.cuh"
#include "jit_lto_kernels/interleaved_scan_planner.hpp"
#include <cstdint>
#include <cuvs/detail/jit_lto/NVRTCLTOFragmentCompiler.hpp>
#include <cuvs/detail/jit_lto/registration_tags.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>

#include "../detail/ann_utils.cuh"
#include <cuvs/distance/distance.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_rt_essentials.hpp>  // RAFT_CUDA_TRY
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cuvs::neighbors::ivf_flat::detail {

static constexpr int kThreadsPerBlock = 128;

using namespace cuvs::spatial::knn::detail;  // NOLINT

// Constexpr mapping functions from actual types to tags
template <typename T>
constexpr auto get_data_type_tag()
{
  if constexpr (std::is_same_v<T, float>) { return tag_f{}; }
  if constexpr (std::is_same_v<T, __half>) { return tag_h{}; }
  if constexpr (std::is_same_v<T, int8_t>) { return tag_sc{}; }
  if constexpr (std::is_same_v<T, uint8_t>) { return tag_uc{}; }
}

template <typename AccT>
constexpr auto get_acc_type_tag()
{
  if constexpr (std::is_same_v<AccT, float>) { return tag_acc_f{}; }
  if constexpr (std::is_same_v<AccT, __half>) { return tag_acc_h{}; }
  if constexpr (std::is_same_v<AccT, int32_t>) { return tag_acc_i{}; }
  if constexpr (std::is_same_v<AccT, uint32_t>) { return tag_acc_ui{}; }
}

template <typename IdxT>
constexpr auto get_idx_type_tag()
{
  if constexpr (std::is_same_v<IdxT, int64_t>) { return tag_idx_l{}; }
}

// Convert type to string for JIT code generation
template <typename T>
constexpr const char* type_name()
{
  if constexpr (std::is_same_v<T, float>) { return "float"; }
  if constexpr (std::is_same_v<T, __half>) { return "__half"; }
  if constexpr (std::is_same_v<T, int8_t>) { return "int8_t"; }
  if constexpr (std::is_same_v<T, uint8_t>) { return "uint8_t"; }
  if constexpr (std::is_same_v<T, int32_t>) { return "int32_t"; }
  if constexpr (std::is_same_v<T, uint32_t>) { return "uint32_t"; }
  if constexpr (std::is_same_v<T, int64_t>) { return "int64_t"; }
}

template <typename FilterT>
constexpr auto get_filter_type_tag()
{
  using namespace cuvs::neighbors::filtering;

  if constexpr (std::is_same_v<FilterT, none_sample_filter>) { return tag_filter_none{}; }
  if constexpr (std::is_same_v<FilterT, bitset_filter<uint32_t, int64_t>>) {
    return tag_filter_bitset{};
  }
}

template <typename MetricTag, int Veclen, typename T, typename AccT>
constexpr auto get_metric_name()
{
  if constexpr (std::is_same_v<MetricTag, tag_metric_euclidean<Veclen, T, AccT>>) {
    return "euclidean";
  } else if constexpr (std::is_same_v<MetricTag, tag_metric_inner_product<Veclen, T, AccT>>) {
    return "inner_prod";
  } else if constexpr (std::is_same_v<MetricTag, tag_metric_custom_udf<Veclen, T, AccT>>) {
    return "metric_udf";
  }
}

template <typename IvfSampleFilterTag>
constexpr auto get_filter_name()
{
  if constexpr (std::is_same_v<IvfSampleFilterTag, tag_filter_none>) {
    return "filter_none_source_index_l";
  }
  if constexpr (std::is_same_v<IvfSampleFilterTag, tag_filter_bitset>) {
    return "filter_bitset_source_index_l";
  }
}

template <typename PostLambdaTag>
constexpr auto get_post_lambda_name()
{
  if constexpr (std::is_same_v<PostLambdaTag, tag_post_identity>) { return "post_identity"; }
  if constexpr (std::is_same_v<PostLambdaTag, tag_post_sqrt>) { return "post_sqrt"; }
  if constexpr (std::is_same_v<PostLambdaTag, tag_post_compose>) { return "post_compose"; }
}

/**
 *  Configure the gridDim.x to maximize GPU occupancy, but reduce the output size
 */
inline uint32_t configure_launch_x(uint32_t numQueries,
                                   uint32_t n_probes,
                                   int32_t sMemSize,
                                   cudaKernel_t func)
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_sms;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, func, kThreadsPerBlock, sMemSize));

  size_t min_grid_size = num_sms * num_blocks_per_sm;
  size_t min_grid_x    = raft::ceildiv<size_t>(min_grid_size, numQueries);
  return min_grid_x > n_probes ? n_probes : static_cast<uint32_t>(min_grid_x);
}

template <int Capacity,
          int Veclen,
          bool Ascending,
          bool ComputeNorm,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterTag,
          typename MetricTag,
          typename PostLambdaTag>
void launch_kernel(const index<T, IdxT>& index,
                   const search_params& params,
                   const T* queries,
                   const uint32_t* coarse_index,
                   const uint32_t num_queries,
                   const uint32_t queries_offset,
                   const uint32_t n_probes,
                   const uint32_t k,
                   const uint32_t max_samples,
                   const uint32_t* chunk_indices,
                   IdxT* const* const inds_ptrs,
                   cuda::std::optional<uint32_t*> bitset_ptr,
                   cuda::std::optional<IdxT> bitset_len,
                   cuda::std::optional<IdxT> original_nbits,
                   uint32_t* neighbors,
                   float* distances,
                   uint32_t& grid_dim_x,
                   rmm::cuda_stream_view stream)
{
  RAFT_EXPECTS(Veclen == index.veclen(),
               "Configured Veclen does not match the index interleaving pattern.");

  // Use tag types for the planner to avoid template bloat
  auto kernel_planner = InterleavedScanPlanner<decltype(get_data_type_tag<T>()),
                                               decltype(get_acc_type_tag<AccT>()),
                                               decltype(get_idx_type_tag<IdxT>())>(
    Capacity, Veclen, Ascending, ComputeNorm);
  if (params.metric_udf.has_value()) {
    std::string metric_udf = params.metric_udf.value();
    // Add explicit template instantiation with actual types
    metric_udf += "\ntemplate void cuvs::neighbors::ivf_flat::detail::compute_dist<";
    metric_udf += std::to_string(Veclen);
    metric_udf += ", ";
    metric_udf += type_name<T>();
    metric_udf += ", ";
    metric_udf += type_name<AccT>();
    metric_udf += ">(";
    metric_udf += type_name<AccT>();
    metric_udf += "&, ";
    metric_udf += type_name<AccT>();
    metric_udf += ", ";
    metric_udf += type_name<AccT>();
    metric_udf += ");\n";
    // Include hash of UDF source in key to differentiate different UDFs
    auto udf_hash            = std::to_string(std::hash<std::string>{}(metric_udf));
    std::string metric_name  = "metric_udf_" + udf_hash;
    auto& nvrtc_lto_compiler = nvrtc_compiler();
    std::string key =
      metric_name + "_" + std::to_string(Veclen) + "_" +
      make_fragment_key<decltype(get_data_type_tag<T>()), decltype(get_acc_type_tag<AccT>())>();
    nvrtc_lto_compiler.compile(key, metric_udf);
    kernel_planner.template add_metric_device_function<decltype(get_data_type_tag<T>()),
                                                       decltype(get_acc_type_tag<AccT>())>(
      metric_name, Veclen);
  } else {
    kernel_planner.template add_metric_device_function<decltype(get_data_type_tag<T>()),
                                                       decltype(get_acc_type_tag<AccT>())>(
      get_metric_name<MetricTag, Veclen, T, AccT>(), Veclen);
  }
  kernel_planner.add_filter_device_function(get_filter_name<IvfSampleFilterTag>());
  kernel_planner.add_post_lambda_device_function(get_post_lambda_name<PostLambdaTag>());
  auto kernel_launcher = kernel_planner.get_launcher();

  const int max_query_smem = 16384;
  int query_smem_elems     = std::min<int>(max_query_smem / sizeof(T),
                                       raft::Pow2<Veclen * raft::WarpSize>::roundUp(index.dim()));
  int smem_size            = query_smem_elems * sizeof(T);

  if constexpr (Capacity > 0) {
    constexpr int kSubwarpSize = std::min<int>(Capacity, raft::WarpSize);
    auto block_merge_mem =
      raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<float, IdxT>(
        kThreadsPerBlock / kSubwarpSize, k);
    smem_size += std::max<int>(smem_size, block_merge_mem);
  }

  // power-of-two less than cuda limit (for better addr alignment)
  constexpr uint32_t kMaxGridY = 32768;

  if (grid_dim_x == 0) {
    grid_dim_x = configure_launch_x(
      std::min(kMaxGridY, num_queries), n_probes, smem_size, kernel_launcher->get_kernel());
    return;
  }

  // Pass individual filter parameters like CAGRA does
  // The kernel will construct filter_data struct internally when needed

  for (uint32_t query_offset = 0; query_offset < num_queries; query_offset += kMaxGridY) {
    uint32_t grid_dim_y = std::min<uint32_t>(kMaxGridY, num_queries - query_offset);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(kThreadsPerBlock);
    RAFT_LOG_TRACE(
      "Launching the ivf-flat interleaved_scan_kernel (%d, %d, 1) x (%d, 1, 1), n_probes = %d, "
      "smem_size = %d",
      grid_dim.x,
      grid_dim.y,
      block_dim.x,
      n_probes,
      smem_size);
    kernel_launcher->dispatch(stream,
                              grid_dim,
                              block_dim,
                              smem_size,
                              query_smem_elems,
                              queries,
                              coarse_index,
                              index.data_ptrs().data_handle(),
                              index.list_sizes().data_handle(),
                              queries_offset + query_offset,
                              n_probes,
                              k,
                              max_samples,
                              chunk_indices,
                              index.dim(),
                              inds_ptrs,
                              bitset_ptr.value_or(nullptr),
                              bitset_len.value_or(0),
                              original_nbits.value_or(0),
                              neighbors,
                              distances);
    queries += grid_dim_y * index.dim();
    if constexpr (Capacity > 0) {
      neighbors += grid_dim_y * grid_dim_x * k;
      distances += grid_dim_y * grid_dim_x * k;
    } else {
      distances += grid_dim_y * max_samples;
    }
    chunk_indices += grid_dim_y * n_probes;
    coarse_index += grid_dim_y * n_probes;
  }
}

/** Select the distance computation function and forward the rest of the arguments. */
template <int Capacity,
          int Veclen,
          bool Ascending,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterTag,
          typename... Args>
void launch_with_fixed_consts(cuvs::distance::DistanceType metric, Args&&... args)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2Expanded:
    case cuvs::distance::DistanceType::L2Unexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_euclidean<Veclen, T, AccT>,
                           tag_post_identity>(std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_euclidean<Veclen, T, AccT>,
                           tag_post_sqrt>(std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::InnerProduct:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_inner_product<Veclen, T, AccT>,
                           tag_post_identity>(std::forward<Args>(args)...);
    case cuvs::distance::DistanceType::CosineExpanded:
      // NB: "Ascending" is reversed because the post-processing step is done after that sort
      return launch_kernel<Capacity,
                           Veclen,
                           !Ascending,
                           true,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_inner_product<Veclen, T, AccT>,
                           tag_post_compose>(
        std::forward<Args>(args)...);  // NB: update the description of `knn::ivf_flat::build` when
                                       // adding here a new metric.
    case cuvs::distance::DistanceType::CustomUDF:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           false,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterTag,
                           tag_metric_custom_udf<Veclen, T, AccT>,
                           tag_post_identity>(std::forward<Args>(args)...);
    default: RAFT_FAIL("The chosen distance metric is not supported (%d)", int(metric));
  }
}

/**
 * Lift the `capacity` and `veclen` parameters to the template level,
 * forward the rest of the arguments unmodified to `launch_interleaved_scan_kernel`.
 */
template <typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterTag,
          int Capacity = raft::matrix::detail::select::warpsort::kMaxCapacity,
          int Veclen   = std::max<int>(1, 16 / sizeof(T))>
struct select_interleaved_scan_kernel {
  /**
   * Recursively reduce the `Capacity` and `Veclen` parameters until they match the
   * corresponding runtime arguments.
   * By default, this recursive process starts with maximum possible values of the
   * two parameters and ends with both values equal to 1.
   */
  template <typename... Args>
  static inline void run(int k_max, int veclen, bool select_min, Args&&... args)
  {
    if constexpr (Capacity > 0) {
      if (k_max == 0 || k_max > Capacity) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterTag, 0, Veclen>::run(
          k_max, veclen, select_min, std::forward<Args>(args)...);
      }
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) {
        return select_interleaved_scan_kernel<T,
                                              AccT,
                                              IdxT,
                                              IvfSampleFilterTag,
                                              Capacity / 2,
                                              Veclen>::run(k_max,
                                                           veclen,
                                                           select_min,
                                                           std::forward<Args>(args)...);
      }
    }
    if constexpr (Veclen > 1) {
      if (veclen % Veclen != 0) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterTag, Capacity, 1>::run(
          k_max, 1, select_min, std::forward<Args>(args)...);
      }
    }
    // NB: this is the limitation of the warpsort structures that use a huge number of
    //     registers (used in the main kernel here).
    RAFT_EXPECTS(Capacity == 0 || k_max == Capacity,
                 "Capacity must be either 0 or a power-of-two not bigger than the maximum "
                 "allowed size matrix::detail::select::warpsort::kMaxCapacity (%d).",
                 raft::matrix::detail::select::warpsort::kMaxCapacity);
    RAFT_EXPECTS(
      veclen == Veclen,
      "Veclen must be power-of-two not bigger than the maximum allowed size for this data type.");
    if (select_min) {
      launch_with_fixed_consts<Capacity, Veclen, true, T, AccT, IdxT, IvfSampleFilterTag>(
        std::forward<Args>(args)...);
    } else {
      launch_with_fixed_consts<Capacity, Veclen, false, T, AccT, IdxT, IvfSampleFilterTag>(
        std::forward<Args>(args)...);
    }
  }
};

/**
 * @brief Configure and launch an appropriate template instance of the interleaved scan kernel.
 *
 * @tparam T value type
 * @tparam AccT accumulated type
 * @tparam IdxT type of the indices
 *
 * @param index previously built ivf-flat index
 * @param[in] queries device pointer to the query vectors [batch_size, dim]
 * @param[in] coarse_query_results device pointer to the cluster (list) ids [batch_size, n_probes]
 * @param n_queries batch size
 * @param[in] queries_offset
 *   An offset of the current query batch. It is used for feeding sample_filter with the
 *   correct query index.
 * @param metric type of the measured distance
 * @param n_probes number of nearest clusters to query
 * @param k number of nearest neighbors.
 *            NB: the maximum value of `k` is limited statically by `kMaxCapacity`.
 * @param select_min whether to select nearest (true) or furthest (false) points w.r.t. the given
 * metric.
 * @param[out] neighbors device pointer to the result indices for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[out] distances device pointer to the result distances for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[inout] grid_dim_x number of blocks launched across all n_probes clusters;
 *               (one block processes one or more probes, hence: 1 <= grid_dim_x <= n_probes)
 * @param stream
 * @param sample_filter
 *   A filter that selects samples for a given query. Use an instance of none_sample_filter to
 *   provide a green light for every sample.
 */
template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void ivfflat_interleaved_scan(const index<T, IdxT>& index,
                              const search_params& params,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const uint32_t queries_offset,
                              const cuvs::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const uint32_t max_samples,
                              const uint32_t* chunk_indices,
                              const bool select_min,
                              IvfSampleFilterT sample_filter,
                              uint32_t* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream)
{
  const int capacity = raft::bound_by_power_of_two(k);

  cuda::std::optional<uint32_t*> bitset_ptr;
  cuda::std::optional<IdxT> bitset_len;
  cuda::std::optional<IdxT> original_nbits;

  if constexpr (std::is_same_v<IvfSampleFilterT,
                               cuvs::neighbors::filtering::bitset_filter<uint32_t, IdxT>>) {
    bitset_ptr     = sample_filter.view().data();
    bitset_len     = sample_filter.view().size();
    original_nbits = sample_filter.view().get_original_nbits();
  }
  select_interleaved_scan_kernel<T, AccT, IdxT, decltype(get_filter_type_tag<IvfSampleFilterT>())>::
    run(capacity,
        index.veclen(),
        select_min,
        metric,
        index,
        params,
        queries,
        coarse_query_results,
        n_queries,
        queries_offset,
        n_probes,
        k,
        max_samples,
        chunk_indices,
        index.inds_ptrs().data_handle(),
        bitset_ptr,
        bitset_len,
        original_nbits,
        neighbors,
        distances,
        grid_dim_x,
        stream);
}

}  // namespace cuvs::neighbors::ivf_flat::detail
