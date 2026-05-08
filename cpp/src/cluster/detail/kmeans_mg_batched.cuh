/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"
#include "kmeans_common.cuh"

#include "../../core/omp_wrapper.hpp"
#include "../../neighbors/detail/ann_utils.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/pinned_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resource/nccl_comm.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <nccl.h>

#include <raft/core/resource/comms.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

template <typename DataT, typename IndexT, typename Accessor>
void sample_rows_from_local_parts(
  raft::resources const& handle,
  const std::vector<
    raft::mdspan<const DataT, raft::matrix_extent<IndexT>, raft::row_major, Accessor>>& X_parts,
  IndexT n_local,
  raft::device_matrix_view<DataT, IndexT> sample,
  uint64_t seed)
{
  using matrix_view_t =
    raft::mdspan<const DataT, raft::matrix_extent<IndexT>, raft::row_major, Accessor>;
  constexpr bool data_on_host   = raft::is_host_mdspan_v<matrix_view_t>;
  constexpr bool data_on_device = raft::is_device_mdspan_v<matrix_view_t>;
  static_assert(data_on_host || data_on_device,
                "partition views must be host- or device-accessible raft mdspans");

  auto stream     = raft::resource::get_cuda_stream(handle);
  auto n_rows     = static_cast<IndexT>(sample.extent(0));
  auto n_features = static_cast<IndexT>(sample.extent(1));

  RAFT_EXPECTS(n_local > 0, "cannot sample centroids from empty local partitions");

  std::vector<IndexT> offsets;
  offsets.reserve(X_parts.size() + 1);
  offsets.push_back(IndexT{0});
  for (auto const& X : X_parts) {
    offsets.push_back(offsets.back() + static_cast<IndexT>(X.extent(0)));
  }

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<IndexT> dist(IndexT{0}, n_local - 1);

  auto choose_row = [&]() {
    IndexT local_index = dist(gen);
    auto upper         = std::upper_bound(offsets.begin(), offsets.end(), local_index);
    auto part_idx      = static_cast<size_t>(std::distance(offsets.begin(), upper) - 1);
    IndexT part_row    = local_index - offsets[part_idx];
    return std::pair<size_t, IndexT>{part_idx, part_row};
  };

  if constexpr (data_on_host) {
    std::vector<DataT> h_sample(static_cast<size_t>(n_rows) * n_features);
    for (IndexT row = 0; row < n_rows; ++row) {
      auto [part_idx, part_row] = choose_row();
      auto const* src           = X_parts[part_idx].data_handle() + part_row * n_features;
      auto* dst                 = h_sample.data() + row * n_features;
      std::copy_n(src, n_features, dst);
    }

    raft::update_device(sample.data_handle(), h_sample.data(), h_sample.size(), stream);
  } else {
    for (IndexT row = 0; row < n_rows; ++row) {
      auto [part_idx, part_row] = choose_row();
      auto const* src           = X_parts[part_idx].data_handle() + part_row * n_features;
      auto* dst                 = sample.data_handle() + row * n_features;
      raft::copy(dst, src, n_features, stream);
    }
  }
}

/**
 * @brief Initialize centroids from this rank's local data partitions.
 *
 * Random samples directly into @p centroids. KMeans++ first copies a subsample
 * of local rows to the device, then seeds centroids from that subsample.
 *
 * @param streaming_batch_size  Upper bound on the KMeans++ subsample size
 *                              (0 disables the cap).
 */
template <typename DataT, typename IndexT, typename Accessor>
void init_centroids_from_part_sample(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& params,
  IndexT streaming_batch_size,
  const std::vector<
    raft::mdspan<const DataT, raft::matrix_extent<IndexT>, raft::row_major, Accessor>>& X_parts,
  IndexT n_local,
  raft::device_matrix_view<DataT, IndexT> centroids,
  rmm::device_uvector<char>& workspace)
{
  auto n_features = static_cast<IndexT>(centroids.extent(1));
  auto n_clusters = static_cast<IndexT>(params.n_clusters);

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    sample_rows_from_local_parts(handle, X_parts, n_local, centroids, params.rng_state.seed);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    IndexT default_init_size = std::min(static_cast<IndexT>(std::int64_t{3} * n_clusters), n_local);
    IndexT init_sample_size  = params.init_size > 0
                                 ? std::min(static_cast<IndexT>(params.init_size), n_local)
                                 : default_init_size;
    if (streaming_batch_size > 0) {
      init_sample_size = std::min(init_sample_size, streaming_batch_size);
    }
    init_sample_size = std::max(init_sample_size, n_clusters);

    auto init_sample =
      raft::make_device_matrix<DataT, IndexT>(handle, init_sample_size, n_features);
    sample_rows_from_local_parts(
      handle, X_parts, n_local, init_sample.view(), params.rng_state.seed);
    auto init_view = raft::make_const_mdspan(init_sample.view());
    if (params.oversampling_factor == 0) {
      kmeansPlusPlus<DataT, IndexT>(handle, params, init_view, centroids, workspace);
    } else {
      initScalableKMeansPlusPlus<DataT, IndexT>(handle, params, init_view, centroids, workspace);
    }
  } else {
    THROW("unknown initialization method to select initial centers");
  }
}

template <typename DataT, typename IndexT>
void init_centroids_from_host_sample(raft::resources const& handle,
                                     const cuvs::cluster::kmeans::params& params,
                                     IndexT streaming_batch_size,
                                     raft::host_matrix_view<const DataT, IndexT> X,
                                     raft::device_matrix_view<DataT, IndexT> centroids,
                                     rmm::device_uvector<char>& workspace)
{
  std::vector<raft::host_matrix_view<const DataT, IndexT>> X_parts{X};
  init_centroids_from_part_sample(handle,
                                  params,
                                  streaming_batch_size,
                                  X_parts,
                                  static_cast<IndexT>(X.extent(0)),
                                  centroids,
                                  workspace);
}

}  // namespace cuvs::cluster::kmeans::detail

namespace cuvs::cluster::kmeans::mg::detail {

// NCCL data-type helper
template <typename T>
ncclDataType_t nccl_dtype();

template <>
inline ncclDataType_t nccl_dtype<float>()
{
  return ncclFloat;
}
template <>
inline ncclDataType_t nccl_dtype<double>()
{
  return ncclDouble;
}
template <>
inline ncclDataType_t nccl_dtype<int64_t>()
{
  return ncclInt64;
}

// Comm macros: dispatch to raw NCCL or RAFT comms based on `use_nccl`.
// Local to this TU, undef'd at the bottom.

#define SNMG_ALLREDUCE(sendbuf, recvbuf, count)                                           \
  do {                                                                                    \
    using _snmg_val_t = std::remove_pointer_t<decltype(sendbuf)>;                         \
    if (use_nccl) {                                                                       \
      RAFT_NCCL_TRY(ncclAllReduce(                                                        \
        sendbuf, recvbuf, count, nccl_dtype<_snmg_val_t>(), ncclSum, nccl_comm, stream)); \
    } else {                                                                              \
      const auto& _snmg_comm = raft::resource::get_comms(dev_res);                        \
      _snmg_comm.allreduce(sendbuf, recvbuf, count, raft::comms::op_t::SUM, stream);      \
    }                                                                                     \
  } while (0)

#define SNMG_BCAST(buf, count, root)                                                           \
  do {                                                                                         \
    using _snmg_bcast_t = std::remove_pointer_t<decltype(buf)>;                                \
    if (use_nccl) {                                                                            \
      RAFT_NCCL_TRY(                                                                           \
        ncclBroadcast(buf, buf, count, nccl_dtype<_snmg_bcast_t>(), root, nccl_comm, stream)); \
    } else {                                                                                   \
      const auto& _snmg_comm = raft::resource::get_comms(dev_res);                             \
      _snmg_comm.bcast(buf, count, root, stream);                                              \
    }                                                                                          \
  } while (0)

#define SNMG_GROUP_START()                             \
  do {                                                 \
    if (use_nccl) { RAFT_NCCL_TRY(ncclGroupStart()); } \
  } while (0)

#define SNMG_GROUP_END()                             \
  do {                                               \
    if (use_nccl) { RAFT_NCCL_TRY(ncclGroupEnd()); } \
  } while (0)

/**
 * @brief Shared multi-GPU k-means fit core, called per rank.
 *
 * Backs both Path 1 (OMP threads sharing a clique with NCCL comms) and Path 2
 * (one rank per process with RAFT comms). The active backend is selected via
 * `raft::resource::is_multi_gpu(handle)`. Each rank streams its local
 * partition list through Lloyd iterations using batched device-side reductions,
 * allreducing partial centroid sums, weights, and clustering cost at the end of each
 * iteration. Best-of-`n_init` is tracked per rank. RAFT comms ranks each write
 * their own caller-provided outputs; SNMG OMP threads share caller-provided
 * outputs, so only rank 0 writes them.
 *
 * @tparam T    Data / weight type (float or double)
 * @tparam IdxT Index type (int or int64_t)
 *
 * @param[in]  handle        RAFT resources. For Path 1 this is the shared
 *                           clique; for Path 2 this is the per-process handle.
 * @param[in]  params        K-means parameters (n_clusters, init, max_iter,
 *                           n_init, tol, metric, batch_*, etc.).
 * @param[in]  X_parts      Local dataset partitions for this rank. May be empty
 *                          on a rank as long as at least one rank has data.
 * @param[in]  sample_weight_parts
 *                          Optional per-partition row weights. When unset, all
 *                          samples are weighted equally.
 * @param[in,out] centroids  Device matrix [n_clusters x n_features]. On entry,
 *                           used as the initial centers when
 *                           `params.init == InitMethod::Array`. On return,
 *                           all RAFT comms ranks write the converged centroids;
 *                           SNMG writes them from rank 0 only.
 * @param[out] inertia       Host scalar receiving the final clustering cost on
 *                           all RAFT comms ranks, or rank 0 for SNMG.
 * @param[out] n_iter        Host scalar receiving the iteration count at which
 *                           the run terminated on all RAFT comms ranks, or
 *                           rank 0 for SNMG.
 * @param[in,out] scaled_weights_cache
 *                           Optional pinned host slice of size n_local used to
 *                           cache this rank's normalized weights so each batch
 *                           is a single async H2D copy. Typically a slice of a
 *                           shared buffer owned by the caller (Path 1); when
 *                           unset a per-rank buffer is allocated internally
 *                           (Path 2). Ignored when @p sample_weight_parts is unset.
 */
template <typename T, typename IdxT, typename MatrixAccessor, typename WeightAccessor>
void mnmg_fit(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::params& params,
  const std::vector<
    raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, MatrixAccessor>>& X_parts,
  const std::optional<std::vector<
    raft::mdspan<const T, raft::vector_extent<IdxT>, raft::layout_right, WeightAccessor>>>&
    sample_weight_parts,
  raft::device_matrix_view<T, IdxT> centroids,
  raft::host_scalar_view<T> inertia,
  raft::host_scalar_view<IdxT> n_iter,
  std::optional<raft::pinned_vector_view<T, IdxT>> scaled_weights_cache = std::nullopt)
{
  // Setup: rank, num_ranks, dev_res, comm mechanism
  bool use_nccl = raft::resource::is_multi_gpu(handle);
  int rank, num_ranks;
  ncclComm_t nccl_comm{};

  if (use_nccl) {
    rank      = cuvs::core::omp::get_thread_num();
    num_ranks = raft::resource::get_num_ranks(handle);
    nccl_comm = raft::resource::get_nccl_comm_for_rank(handle, rank);
  } else {
    const auto& comm = raft::resource::get_comms(handle);
    rank             = comm.get_rank();
    num_ranks        = comm.get_size();
  }

  const raft::resources& dev_res =
    use_nccl ? raft::resource::set_current_device_to_rank(handle, rank) : handle;

  using matrix_view_t =
    raft::mdspan<const T, raft::matrix_extent<IdxT>, raft::row_major, MatrixAccessor>;
  using weight_view_t =
    raft::mdspan<const T, raft::vector_extent<IdxT>, raft::layout_right, WeightAccessor>;
  constexpr bool data_on_host      = raft::is_host_mdspan_v<matrix_view_t>;
  constexpr bool data_on_device    = raft::is_device_mdspan_v<matrix_view_t>;
  constexpr bool weights_on_host   = raft::is_host_mdspan_v<weight_view_t>;
  constexpr bool weights_on_device = raft::is_device_mdspan_v<weight_view_t>;
  static_assert(data_on_host || data_on_device,
                "partition views must be host- or device-accessible raft mdspans");
  static_assert(weights_on_host || weights_on_device,
                "weight views must be host- or device-accessible raft mdspans");

  auto stream     = raft::resource::get_cuda_stream(dev_res);
  auto n_features = static_cast<IdxT>(centroids.extent(1));
  auto n_clusters = static_cast<IdxT>(params.n_clusters);
  auto metric     = params.metric;

  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IdxT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(n_features > 0, "centroids.extent(1) must be positive");
  RAFT_EXPECTS(num_ranks > 0, "num_ranks must be positive");

  IdxT n_local       = 0;
  IdxT max_part_rows = 0;
  std::vector<IdxT> part_offsets;
  part_offsets.reserve(X_parts.size() + 1);
  part_offsets.push_back(IdxT{0});
  for (auto const& X_part : X_parts) {
    RAFT_EXPECTS(static_cast<IdxT>(X_part.extent(1)) == n_features,
                 "all partitions must have the same feature count as centroids");
    auto part_rows = static_cast<IdxT>(X_part.extent(0));
    n_local += part_rows;
    max_part_rows = std::max(max_part_rows, part_rows);
    part_offsets.push_back(n_local);
  }

  if (sample_weight_parts.has_value()) {
    RAFT_EXPECTS(sample_weight_parts->size() == X_parts.size(),
                 "sample_weight_parts must have one entry per data partition");
    for (size_t i = 0; i < X_parts.size(); ++i) {
      RAFT_EXPECTS(static_cast<IdxT>((*sample_weight_parts)[i].extent(0)) ==
                     static_cast<IdxT>(X_parts[i].extent(0)),
                   "each sample_weight partition must match its X partition rows");
    }
  }

  auto d_global_n = raft::make_device_scalar<IdxT>(dev_res, n_local);
  SNMG_ALLREDUCE(d_global_n.data_handle(), d_global_n.data_handle(), 1);
  IdxT global_n{};
  raft::copy(&global_n, d_global_n.data_handle(), 1, stream);
  raft::resource::sync_stream(dev_res);
  RAFT_EXPECTS(global_n > 0, "at least one sample is required across all ranks");

  RAFT_LOG_DEBUG("SNMG KMeans fit: rank=%d/%d, n_local=%zu, n_features=%zu, n_clusters=%d",
                 rank,
                 num_ranks,
                 static_cast<size_t>(n_local),
                 static_cast<size_t>(n_features),
                 static_cast<int>(n_clusters));

  IdxT streaming_batch_size = static_cast<IdxT>(params.streaming_batch_size);
  if (streaming_batch_size <= 0 || streaming_batch_size > max_part_rows) {
    streaming_batch_size = std::max(max_part_rows, IdxT{1});
  }

  bool has_data = (n_local > 0);

  // Work buffers, allocated once and reused across iterations
  auto rank_centroids        = raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features);
  auto new_centroids         = raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features);
  auto centroid_sums         = raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features);
  auto weight_per_cluster    = raft::make_device_vector<T, IdxT>(dev_res, n_clusters);
  auto clustering_cost       = raft::make_device_vector<T, IdxT>(dev_res, 1);
  auto batch_clustering_cost = raft::make_device_vector<T, IdxT>(dev_res, 1);
  auto sqrd_norm_error_dev   = raft::make_device_scalar<T>(dev_res, T{0});
  IdxT alloc_batch_size      = has_data ? streaming_batch_size : IdxT{1};
  auto batch_weights         = raft::make_device_vector<T, IdxT>(dev_res, alloc_batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IdxT, T>, IdxT>(dev_res, alloc_batch_size);
  auto L2NormBatch = raft::make_device_vector<T, IdxT>(dev_res, alloc_batch_size);
  rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_uvector<char> workspace(0, stream);
  rmm::device_uvector<char> batch_workspace(0, stream);

  // Weight normalization: rescale so global weights sum to global_n.
  T weight_scale = T{1};
  if (sample_weight_parts.has_value()) {
    auto d_wt = raft::make_device_scalar<T>(dev_res, T{0});
    for (auto const& weights : sample_weight_parts.value()) {
      auto n_weights = static_cast<IdxT>(weights.extent(0));
      if (n_weights == 0) { continue; }

      auto d_part_wt = raft::make_device_scalar<T>(dev_res, T{0});
      cuvs::cluster::kmeans::detail::weightSum(dev_res, weights, d_part_wt.view());
      raft::linalg::add(d_wt.data_handle(), d_wt.data_handle(), d_part_wt.data_handle(), 1, stream);
    }

    SNMG_ALLREDUCE(d_wt.data_handle(), d_wt.data_handle(), 1);

    T global_wt{};
    raft::copy(&global_wt, d_wt.data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);

    RAFT_EXPECTS(std::isfinite(global_wt) && global_wt > T{0},
                 "invalid parameter (sum of sample weights must be finite and positive)");
    const auto global_n_wt = static_cast<T>(global_n);
    const T tol            = global_n_wt * std::numeric_limits<T>::epsilon();
    if (std::abs(global_wt - global_n_wt) > tol) { weight_scale = global_n_wt / global_wt; }
  }

  // Cache the rescaled weights once in pinned host memory so each batch is a
  // single async H2D copy with no follow-up rescale kernel.
  std::optional<raft::pinned_vector<T, IdxT>> local_scaled_weights;
  T* scaled_weights_ptr = nullptr;
  if (sample_weight_parts.has_value()) {
    if constexpr (weights_on_host) {
      if (scaled_weights_cache.has_value()) {
        RAFT_EXPECTS(static_cast<IdxT>(scaled_weights_cache->extent(0)) == n_local,
                     "scaled_weights_cache must have extent equal to n_local");
        scaled_weights_ptr = scaled_weights_cache->data_handle();
      } else {
        local_scaled_weights = raft::make_pinned_vector<T, IdxT>(dev_res, n_local);
        scaled_weights_ptr   = local_scaled_weights->data_handle();
      }

      for (size_t part_idx = 0; part_idx < sample_weight_parts->size(); ++part_idx) {
        auto const& weights = (*sample_weight_parts)[part_idx];
        auto part_rows      = static_cast<IdxT>(weights.extent(0));
        auto* dst           = scaled_weights_ptr + part_offsets[part_idx];
        auto const* src     = weights.data_handle();
        for (IdxT i = 0; i < part_rows; ++i) {
          dst[i] = src[i] * weight_scale;
        }
      }
    } else {
      RAFT_EXPECTS(!scaled_weights_cache.has_value(),
                   "scaled_weights_cache is only supported with host sample weights");
    }
  }

  auto n_init = params.n_init;
  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array && n_init != 1) {
    RAFT_LOG_DEBUG(
      "Explicit initial center position passed: performing only one init in "
      "k-means instead of n_init=%d",
      n_init);
    n_init = 1;
  }

  auto best_centroids = n_init > 1
                          ? raft::make_device_matrix<T, IdxT>(dev_res, n_clusters, n_features)
                          : raft::make_device_matrix<T, IdxT>(dev_res, 0, 0);
  T best_inertia      = std::numeric_limits<T>::max();
  IdxT best_n_iter    = 0;

  // Per-rank state (avoid races on shared host scalars in OMP)
  T local_inertia   = T{0};
  IdxT local_n_iter = 0;

  std::mt19937 gen(params.rng_state.seed);

  // On-device convergence state, mirroring single-GPU `detail::fit`. All
  // inputs to `check_convergence` are globally identical across ranks
  // (clustering_cost is allreduced; centroids are derived from allreduced
  // sums/weights), so the flag is computed locally on each rank with no
  // cross-rank reduction. The pinned-host shadow is consumed at the top of
  // the next iteration so the D2H copy can overlap with queued device work.
  auto d_prior_cost = raft::make_device_scalar<T>(dev_res, T{0});
  auto d_done_flag  = raft::make_device_scalar<int64_t>(dev_res, 0);
  auto h_done_flag  = raft::make_pinned_scalar<int64_t>(dev_res, 0);

  bool need_compute_norms = metric == cuvs::distance::DistanceType::L2Expanded ||
                            metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                            metric == cuvs::distance::DistanceType::CosineExpanded;
  auto h_norm_cache =
    raft::make_pinned_vector<T, IdxT>(dev_res, (need_compute_norms && has_data) ? n_local : 0);
  bool norms_cached = false;

  auto prepare_batch_weights = [&](size_t part_idx, IdxT batch_offset, IdxT cur_batch_size) {
    if (sample_weight_parts.has_value()) {
      if (scaled_weights_ptr != nullptr) {
        raft::copy(batch_weights.data_handle(),
                   scaled_weights_ptr + part_offsets[part_idx] + batch_offset,
                   cur_batch_size,
                   stream);
      } else {
        raft::copy(batch_weights.data_handle(),
                   (*sample_weight_parts)[part_idx].data_handle() + batch_offset,
                   cur_batch_size,
                   stream);
        if (weight_scale != T{1}) {
          auto bw =
            raft::make_device_vector_view<T, IdxT>(batch_weights.data_handle(), cur_batch_size);
          raft::linalg::map(
            dev_res, bw, raft::mul_const_op<T>{weight_scale}, raft::make_const_mdspan(bw));
        }
      }
    }

    return raft::make_device_vector_view<const T, IdxT>(batch_weights.data_handle(),
                                                        cur_batch_size);
  };

  auto for_each_local_batch = [&](auto&& fn) {
    for (size_t part_idx = 0; part_idx < X_parts.size(); ++part_idx) {
      auto const& X_part = X_parts[part_idx];
      auto part_rows     = static_cast<IdxT>(X_part.extent(0));
      if (part_rows == 0) { continue; }

      cuvs::spatial::knn::detail::utils::batch_load_iterator<T> data_batches(
        X_part.data_handle(),
        static_cast<size_t>(part_rows),
        static_cast<size_t>(n_features),
        static_cast<size_t>(streaming_batch_size),
        stream,
        rmm::mr::get_current_device_resource_ref(),
        true);
      data_batches.prefetch_next_batch();

      for (auto const& data_batch : data_batches) {
        fn(part_idx, data_batch);
        data_batches.prefetch_next_batch();
      }
    }
  };

  for (int seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    cuvs::cluster::kmeans::params iter_params = params;
    iter_params.rng_state.seed                = gen();

    // Centroid init: rank 0 produces, then broadcast to all ranks
    if (iter_params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
      if (rank == 0) {
        cuvs::cluster::kmeans::detail::init_centroids_from_part_sample<T, IdxT>(
          dev_res,
          iter_params,
          streaming_batch_size,
          X_parts,
          n_local,
          rank_centroids.view(),
          workspace);
      }
    } else {
      if (rank == 0) {
        raft::copy(
          rank_centroids.data_handle(), centroids.data_handle(), n_clusters * n_features, stream);
      }
    }
    SNMG_BCAST(rank_centroids.data_handle(), n_clusters * n_features, 0);

    if (has_data && !sample_weight_parts.has_value()) {
      raft::matrix::fill(dev_res, batch_weights.view(), T{1});
    }

    // Reset per-pass convergence state to avoid leaking it across n_init.
    raft::matrix::fill(dev_res, d_prior_cost.view(), T{0});
    *h_done_flag.data_handle() = 0;

    for (local_n_iter = 1; local_n_iter <= iter_params.max_iter; ++local_n_iter) {
      // Consume the previous iteration's done flag from pinned host.
      if (local_n_iter > 1) {
        raft::resource::sync_stream(dev_res);
        if (*h_done_flag.data_handle()) {
          --local_n_iter;
          RAFT_LOG_DEBUG("SNMG KMeans: threshold triggered after %d iterations on rank %d",
                         static_cast<int>(local_n_iter),
                         rank);
          break;
        }
      }

      RAFT_LOG_DEBUG("SNMG KMeans: iteration %d on rank %d", local_n_iter, rank);

      raft::matrix::fill(dev_res, centroid_sums.view(), T{0});
      raft::matrix::fill(dev_res, weight_per_cluster.view(), T{0});
      raft::matrix::fill(dev_res, clustering_cost.view(), T{0});

      auto rank_centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        rank_centroids.data_handle(), n_clusters, n_features);

      // Phase 1: local batch accumulation (skip if no local data)
      if (has_data) {
        for_each_local_batch([&](size_t part_idx, auto const& data_batch) {
          IdxT current_batch_size = static_cast<IdxT>(data_batch.size());
          auto batch_offset       = static_cast<IdxT>(data_batch.offset());

          auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
            data_batch.data(), current_batch_size, n_features);

          auto batch_weights_view =
            prepare_batch_weights(part_idx, batch_offset, current_batch_size);

          auto L2NormBatch_view =
            raft::make_device_vector_view<T, IdxT>(L2NormBatch.data_handle(), current_batch_size);

          if (need_compute_norms) {
            auto norm_offset = part_offsets[part_idx] + batch_offset;
            if (!norms_cached) {
              if (metric == cuvs::distance::DistanceType::CosineExpanded) {
                raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                  dev_res, batch_data_view, L2NormBatch_view, raft::sqrt_op{});
              } else {
                raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                  dev_res, batch_data_view, L2NormBatch_view);
              }
              raft::copy(h_norm_cache.data_handle() + norm_offset,
                         L2NormBatch.data_handle(),
                         current_batch_size,
                         stream);
            } else {
              raft::copy(L2NormBatch.data_handle(),
                         h_norm_cache.data_handle() + norm_offset,
                         current_batch_size,
                         stream);
            }
          }

          auto L2NormBatch_const = raft::make_const_mdspan(L2NormBatch_view);

          auto minClusterAndDistance_view =
            raft::make_device_vector_view<raft::KeyValuePair<IdxT, T>, IdxT>(
              minClusterAndDistance.data_handle(), current_batch_size);

          cuvs::cluster::kmeans::detail::process_batch<T, IdxT>(
            dev_res,
            batch_data_view,
            batch_weights_view,
            rank_centroids_const,
            metric,
            params.batch_samples,
            params.batch_centroids,
            minClusterAndDistance_view,
            L2NormBatch_const,
            L2NormBuf_OR_DistBuf,
            workspace,
            centroid_sums.view(),
            weight_per_cluster.view(),
            raft::make_device_scalar_view(clustering_cost.data_handle()),
            batch_workspace);
        });
        if (need_compute_norms) { norms_cached = true; }
      }

      // Phase 2: grouped allreduce
      SNMG_GROUP_START();
      SNMG_ALLREDUCE(
        centroid_sums.data_handle(), centroid_sums.data_handle(), n_clusters * n_features);
      SNMG_ALLREDUCE(
        weight_per_cluster.data_handle(), weight_per_cluster.data_handle(), n_clusters);
      SNMG_ALLREDUCE(clustering_cost.data_handle(), clustering_cost.data_handle(), 1);
      SNMG_GROUP_END();

      // Phase 3: finalize centroids
      auto centroid_sums_const = raft::make_device_matrix_view<const T, IdxT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto weight_per_cluster_const =
        raft::make_device_vector_view<const T, IdxT>(weight_per_cluster.data_handle(), n_clusters);

      cuvs::cluster::kmeans::detail::finalize_centroids<T, IdxT>(dev_res,
                                                                 centroid_sums_const,
                                                                 weight_per_cluster_const,
                                                                 rank_centroids_const,
                                                                 new_centroids.view());

      // Phase 4: device-side convergence evaluation. Compute shift,
      // run `check_convergence` via `map_offset`, shadow into pinned host.
      // Consumed at top of next iteration.
      cuvs::cluster::kmeans::detail::compute_centroid_shift<T, IdxT>(
        dev_res,
        raft::make_const_mdspan(rank_centroids.view()),
        raft::make_const_mdspan(new_centroids.view()),
        sqrd_norm_error_dev.view());

      raft::copy(
        rank_centroids.data_handle(), new_centroids.data_handle(), n_clusters * n_features, stream);

      auto d_cost_view  = raft::make_device_scalar_view<const T>(clustering_cost.data_handle());
      auto d_prior_view = d_prior_cost.view();
      auto d_norm_view  = raft::make_device_scalar_view<const T>(sqrd_norm_error_dev.data_handle());
      auto d_done_view  = d_done_flag.view();
      T tol             = static_cast<T>(params.tol);
      int iter          = static_cast<int>(local_n_iter);

      raft::linalg::map_offset(
        dev_res,
        raft::make_device_vector_view<int64_t, int>(d_done_flag.data_handle(), 1),
        [=] __device__(int) {
          cuvs::cluster::kmeans::detail::check_convergence(
            d_cost_view, d_prior_view, d_norm_view, tol, iter, d_done_view);
          return *d_done_view.data_handle();
        });

      raft::copy(dev_res,
                 raft::make_pinned_scalar_view(h_done_flag.data_handle()),
                 raft::make_device_scalar_view<const int64_t>(d_done_flag.data_handle()));
    }

    // Recompute inertia against the converged centroids
    raft::matrix::fill(dev_res, clustering_cost.view(), T{0});
    if (has_data) {
      auto rank_centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        rank_centroids.data_handle(), n_clusters, n_features);

      for_each_local_batch([&](size_t part_idx, auto const& data_batch) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());
        auto batch_offset       = static_cast<IdxT>(data_batch.offset());

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        std::optional<raft::device_vector_view<const T, IdxT>> batch_sw = std::nullopt;
        if (sample_weight_parts.has_value()) {
          batch_sw = prepare_batch_weights(part_idx, batch_offset, current_batch_size);
        }

        raft::matrix::fill(dev_res, batch_clustering_cost.view(), T{0});
        cuvs::cluster::kmeans::cluster_cost(
          dev_res,
          batch_data_view,
          rank_centroids_const,
          raft::make_device_scalar_view(batch_clustering_cost.data_handle()),
          batch_sw);

        raft::linalg::add(dev_res,
                          raft::make_const_mdspan(clustering_cost.view()),
                          raft::make_const_mdspan(batch_clustering_cost.view()),
                          clustering_cost.view());
      });
    }
    SNMG_ALLREDUCE(clustering_cost.data_handle(), clustering_cost.data_handle(), 1);
    raft::copy(&local_inertia, clustering_cost.data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);

    RAFT_LOG_DEBUG("SNMG KMeans: n_init %d/%d completed, inertia=%f, n_iter=%d on rank %d",
                   seed_iter + 1,
                   n_init,
                   static_cast<double>(local_inertia),
                   local_n_iter,
                   rank);

    if (n_init > 1 && local_inertia < best_inertia) {
      best_inertia = local_inertia;
      best_n_iter  = local_n_iter;
      raft::copy(best_centroids.data_handle(),
                 rank_centroids.data_handle(),
                 n_clusters * n_features,
                 stream);
    }
  }

  // Final output: RAFT comms ranks are separate processes with separate output views.
  // SNMG ranks are OMP threads sharing the caller outputs, so only rank 0 writes.
  if (n_init > 1) {
    raft::copy(
      rank_centroids.data_handle(), best_centroids.data_handle(), n_clusters * n_features, stream);
    local_inertia = best_inertia;
    local_n_iter  = best_n_iter;
  }

  bool write_outputs = !use_nccl || rank == 0;
  if (write_outputs) {
    raft::copy(
      centroids.data_handle(), rank_centroids.data_handle(), n_clusters * n_features, stream);
    inertia[0] = local_inertia;
    n_iter[0]  = local_n_iter;
  }
}

template <typename T, typename IdxT>
void mnmg_fit(const raft::resources& handle,
              const cuvs::cluster::kmeans::params& params,
              raft::host_matrix_view<const T, IdxT> X_local,
              std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
              raft::device_matrix_view<T, IdxT> centroids,
              raft::host_scalar_view<T> inertia,
              raft::host_scalar_view<IdxT> n_iter,
              std::optional<raft::pinned_vector_view<T, IdxT>> scaled_weights_cache = std::nullopt)
{
  std::vector<raft::host_matrix_view<const T, IdxT>> X_parts{X_local};

  std::optional<std::vector<raft::host_vector_view<const T, IdxT>>> sample_weight_parts =
    std::nullopt;
  if (sample_weight.has_value()) {
    sample_weight_parts.emplace(
      std::vector<raft::host_vector_view<const T, IdxT>>{sample_weight.value()});
  }

  mnmg_fit<T, IdxT>(
    handle, params, X_parts, sample_weight_parts, centroids, inertia, n_iter, scaled_weights_cache);
}

// OpenMP wrapper for Path 1: one rank per GPU within a single process.
template <typename T, typename IdxT>
void batched_fit_omp(const raft::resources& clique,
                     const cuvs::cluster::kmeans::params& params,
                     raft::host_matrix_view<const T, IdxT> X,
                     std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
                     raft::device_matrix_view<T, IdxT> centroids,
                     raft::host_scalar_view<T> inertia,
                     raft::host_scalar_view<IdxT> n_iter)
{
  raft::resource::get_nccl_comms(clique);
  int num_ranks   = raft::resource::get_num_ranks(clique);
  IdxT n_samples  = X.extent(0);
  IdxT n_features = X.extent(1);

  // Shared pinned cache for rescaled weights; each rank writes its own
  // disjoint slice, so no inter-rank synchronization is needed.
  std::optional<raft::pinned_vector<T, IdxT>> scaled_weights_cache;
  if (sample_weight.has_value()) {
    scaled_weights_cache = raft::make_pinned_vector<T, IdxT>(clique, n_samples);
  }

  IdxT base = n_samples / num_ranks;
  IdxT rem  = n_samples % num_ranks;

  cuvs::core::omp::check_threads(num_ranks);
#pragma omp parallel num_threads(num_ranks)
  {
    int r        = cuvs::core::omp::get_thread_num();
    IdxT offset  = r * base + std::min<IdxT>(r, rem);
    IdxT n_local = base + (r < rem ? 1 : 0);

    auto X_local = raft::make_host_matrix_view<const T, IdxT>(
      X.data_handle() + offset * n_features, n_local, n_features);

    std::optional<raft::host_vector_view<const T, IdxT>> sw_local;
    std::optional<raft::pinned_vector_view<T, IdxT>> sw_local_cache;
    if (sample_weight.has_value()) {
      sw_local =
        raft::make_host_vector_view<const T, IdxT>(sample_weight->data_handle() + offset, n_local);
      sw_local_cache = raft::make_pinned_vector_view<T, IdxT>(
        scaled_weights_cache->data_handle() + offset, n_local);
    }

    mnmg_fit<T, IdxT>(
      clique, params, X_local, sw_local, centroids, inertia, n_iter, sw_local_cache);
  }
}

// Undef local macros
#undef SNMG_ALLREDUCE
#undef SNMG_BCAST
#undef SNMG_GROUP_START
#undef SNMG_GROUP_END

}  // namespace cuvs::cluster::kmeans::mg::detail
