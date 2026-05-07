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
#include <random>
#include <type_traits>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Initialize centroids from a host data matrix.
 *
 * Implements the Random and KMeansPlusPlus initialization paths for k-means
 * when the input dataset resides on the host. For KMeans++ we first copy a
 * subsample of host rows to the device and then seed centroids from that
 * subsample; for Random we sample directly into the output centroids.
 *
 * @tparam DataT  Data type
 * @tparam IndexT Index type
 *
 * @param[in]    handle                  RAFT resources handle
 * @param[in]    params                  Parameters (uses init, rng_state, n_clusters,
 *                                       metric, oversampling_factor, init_size)
 * @param[in]    streaming_batch_size    Upper bound on KMeans++ subsample size
 *                                       (use 0 to disable the cap)
 * @param[in]    X                       Host data [n_samples x n_features]
 * @param[out]   centroids               Output centroids [n_clusters x n_features]
 * @param[inout] workspace               Resizable scratch
 */
template <typename DataT, typename IndexT>
void init_centroids_from_host_sample(raft::resources const& handle,
                                     const cuvs::cluster::kmeans::params& params,
                                     IndexT streaming_batch_size,
                                     raft::host_matrix_view<const DataT, IndexT> X,
                                     raft::device_matrix_view<DataT, IndexT> centroids,
                                     rmm::device_uvector<char>& workspace)
{
  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = static_cast<IndexT>(params.n_clusters);
  raft::random::RngState rng(params.rng_state.seed);

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    raft::matrix::sample_rows(handle, rng, X, centroids);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    IndexT default_init_size =
      std::min(static_cast<IndexT>(std::int64_t{3} * n_clusters), n_samples);
    IndexT init_sample_size = params.init_size > 0
                                ? std::min(static_cast<IndexT>(params.init_size), n_samples)
                                : default_init_size;
    if (streaming_batch_size > 0) {
      init_sample_size = std::min(init_sample_size, streaming_batch_size);
    }
    init_sample_size = std::max(init_sample_size, std::min(n_clusters, n_samples));

    auto init_sample =
      raft::make_device_matrix<DataT, IndexT>(handle, init_sample_size, n_features);
    raft::matrix::sample_rows(handle, rng, X, init_sample.view());
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

/**
 * @brief Copy a slice of host sample weights to a device buffer and scale.
 *
 * No-op when @p sample_weight is std::nullopt (the caller is expected to have
 * pre-filled @p batch_weights with 1's in that case).
 *
 * @tparam DataT  Data type
 * @tparam IndexT Index type
 *
 * @param[in]  handle         RAFT resources handle
 * @param[in]  sample_weight  Optional full host weight vector
 * @param[in]  batch_offset   Offset into the host weights for this batch
 * @param[in]  batch_size     Number of weights to copy
 * @param[in]  weight_scale   Multiplicative scale factor applied after copy
 * @param[out] batch_weights  Device-side storage for the batch weights
 *                            (only the first @p batch_size elements are written)
 */
template <typename DataT, typename IndexT>
void copy_and_scale_batch_weights(
  raft::resources const& handle,
  std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
  IndexT batch_offset,
  IndexT batch_size,
  DataT weight_scale,
  raft::device_vector_view<DataT, IndexT> batch_weights)
{
  if (!sample_weight.has_value()) return;
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::copy(
    batch_weights.data_handle(), sample_weight->data_handle() + batch_offset, batch_size, stream);
  if (weight_scale != DataT{1}) {
    auto bw = raft::make_device_vector_view<DataT, IndexT>(batch_weights.data_handle(), batch_size);
    raft::linalg::map(
      handle,
      bw,
      [weight_scale] __device__(DataT w) { return w * weight_scale; },
      raft::make_const_mdspan(bw));
  }
}

/**
 * @brief Accumulate this batch's weighted centroid sums and weight totals.
 *
 * Adds the per-cluster weighted sums and weight totals computed for this
 * batch into the running @p centroid_sums and @p weight_per_cluster
 * accumulators. Internally a single fused reduce is performed via
 * @ref compute_centroid_adjustments with `reset_sums=false`.
 *
 * @tparam DataT  Data type
 * @tparam IndexT Index type
 */
template <typename DataT, typename IndexT>
void accumulate_batch_centroids(raft::resources const& handle,
                                raft::device_matrix_view<const DataT, IndexT> batch_data,
                                raft::device_vector_view<const raft::KeyValuePair<IndexT, DataT>,
                                                         IndexT> min_cluster_and_distance,
                                raft::device_vector_view<const DataT, IndexT> batch_weights,
                                raft::device_matrix_view<DataT, IndexT> centroid_sums,
                                raft::device_vector_view<DataT, IndexT> weight_per_cluster)
{
  auto stream    = raft::resource::get_cuda_stream(handle);
  auto n_samples = batch_data.extent(0);
  rmm::device_uvector<char> batch_workspace(n_samples, stream);

  KeyValueIndexOp<IndexT, DataT> conversion_op;
  thrust::transform_iterator<KeyValueIndexOp<IndexT, DataT>,
                             const raft::KeyValuePair<IndexT, DataT>*>
    labels_itr(min_cluster_and_distance.data_handle(), conversion_op);

  compute_centroid_adjustments(handle,
                               batch_data,
                               batch_weights,
                               labels_itr,
                               static_cast<IndexT>(centroid_sums.extent(0)),
                               centroid_sums,
                               weight_per_cluster,
                               batch_workspace,
                               /*reset_sums=*/false);
}

}  // namespace cuvs::cluster::kmeans::detail

namespace cuvs::cluster::kmeans::mg::detail {

// ---------------------------------------------------------------------------
// NCCL data-type helper
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Comm macros — select raw NCCL vs RAFT comms at each call site.
// These are local to this translation unit and undef'd at the bottom.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// mnmg_fit — shared multi-GPU core (Paths 1 & 2)
// ---------------------------------------------------------------------------
template <typename T, typename IdxT>
void mnmg_fit(const raft::resources& handle,
              const cuvs::cluster::kmeans::params& params,
              raft::host_matrix_view<const T, IdxT> X_local,
              std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
              raft::device_matrix_view<T, IdxT> centroids,
              raft::host_scalar_view<T> inertia,
              raft::host_scalar_view<IdxT> n_iter)
{
  // --- Setup: rank, num_ranks, dev_res, comm mechanism ---
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

  auto stream     = raft::resource::get_cuda_stream(dev_res);
  auto n_local    = X_local.extent(0);
  auto n_features = X_local.extent(1);
  auto n_clusters = static_cast<IdxT>(params.n_clusters);
  auto metric     = params.metric;

  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IdxT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");
  RAFT_EXPECTS(num_ranks > 0, "num_ranks must be positive");

  RAFT_LOG_DEBUG("SNMG KMeans fit: rank=%d/%d, n_local=%zu, n_features=%zu, n_clusters=%d",
                 rank,
                 num_ranks,
                 static_cast<size_t>(n_local),
                 static_cast<size_t>(n_features),
                 static_cast<int>(n_clusters));

  // --- Resolve streaming batch size ---
  IdxT streaming_batch_size = static_cast<IdxT>(params.streaming_batch_size);
  if (streaming_batch_size <= 0 || streaming_batch_size > n_local) {
    streaming_batch_size = std::max(n_local, IdxT{1});
  }

  bool has_data = (n_local > 0);

  // --- Allocate work buffers once (O2) ---
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

  // --- Weight normalization via allreduce (only when sample weights are provided) ---
  T weight_scale = T{1};
  if (sample_weight.has_value()) {
    auto d_n_local = raft::make_device_scalar<IdxT>(dev_res, static_cast<IdxT>(n_local));
    SNMG_ALLREDUCE(d_n_local.data_handle(), d_n_local.data_handle(), 1);
    raft::resource::sync_stream(dev_res);
    IdxT global_n{};
    raft::copy(&global_n, d_n_local.data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);

    T local_wt_sum = T{0};
    const T* sw    = sample_weight->data_handle();
    for (IdxT i = 0; i < n_local; ++i) {
      local_wt_sum += sw[i];
    }
    auto d_wt = raft::make_device_scalar<T>(dev_res, local_wt_sum);
    SNMG_ALLREDUCE(d_wt.data_handle(), d_wt.data_handle(), 1);
    raft::resource::sync_stream(dev_res);
    T global_wt{};
    raft::copy(&global_wt, d_wt.data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);
    RAFT_EXPECTS(std::isfinite(global_wt) && global_wt > T{0},
                 "invalid parameter (sum of sample weights must be finite and positive)");
    const auto global_n_wt = static_cast<T>(global_n);
    const T tol            = global_n_wt * std::numeric_limits<T>::epsilon();
    if (std::abs(global_wt - global_n_wt) > tol) { weight_scale = global_n_wt / global_wt; }
  }

  // --- n_init handling ---
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

  // Per-rank local state (avoids data races on shared host scalars in OMP)
  T local_inertia   = T{0};
  IdxT local_n_iter = 0;

  std::mt19937 gen(params.rng_state.seed);

  // Allreduce scratch for synchronized convergence
  auto d_done = raft::make_device_scalar<int64_t>(dev_res, 0);

  // Construct the batch iterator once; reset it each Lloyd iter / n_init iter.
  std::optional<cuvs::spatial::knn::detail::utils::batch_load_iterator<T>> data_batches_opt;
  if (has_data) {
    data_batches_opt.emplace(X_local.data_handle(),
                             n_local,
                             n_features,
                             streaming_batch_size,
                             stream,
                             rmm::mr::get_current_device_resource_ref(),
                             true);
  }

  // --- Main n_init loop ---
  for (int seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    cuvs::cluster::kmeans::params iter_params = params;
    iter_params.rng_state.seed                = gen();

    // --- Centroid initialization (rank 0 only, then broadcast) ---
    if (iter_params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
      if (rank == 0) {
        cuvs::cluster::kmeans::detail::init_centroids_from_host_sample<T, IdxT>(
          dev_res, iter_params, streaming_batch_size, X_local, rank_centroids.view(), workspace);
      }
    } else {
      if (rank == 0) {
        raft::copy(
          rank_centroids.data_handle(), centroids.data_handle(), n_clusters * n_features, stream);
      }
    }
    raft::resource::sync_stream(dev_res);
    SNMG_BCAST(rank_centroids.data_handle(), n_clusters * n_features, 0);
    raft::resource::sync_stream(dev_res);

    if (has_data && !sample_weight.has_value()) {
      raft::matrix::fill(dev_res, batch_weights.view(), T{1});
    }

    T prior_cluster_cost = T{0};

    // --- Lloyd iterations ---
    for (local_n_iter = 1; local_n_iter <= iter_params.max_iter; ++local_n_iter) {
      RAFT_LOG_DEBUG("SNMG KMeans: iteration %d on rank %d", local_n_iter, rank);

      raft::matrix::fill(dev_res, centroid_sums.view(), T{0});
      raft::matrix::fill(dev_res, weight_per_cluster.view(), T{0});
      raft::matrix::fill(dev_res, clustering_cost.view(), T{0});

      auto rank_centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        rank_centroids.data_handle(), n_clusters, n_features);

      // Phase 1: local batch accumulation (skip if no local data)
      if (has_data) {
        auto& data_batches = *data_batches_opt;
        data_batches.reset();
        for (const auto& data_batch : data_batches) {
          IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

          raft::matrix::fill(dev_res, batch_clustering_cost.view(), T{0});

          auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
            data_batch.data(), current_batch_size, n_features);

          cuvs::cluster::kmeans::detail::copy_and_scale_batch_weights<T, IdxT>(
            dev_res,
            sample_weight,
            static_cast<IdxT>(data_batch.offset()),
            current_batch_size,
            weight_scale,
            batch_weights.view());

          auto batch_weights_view = raft::make_device_vector_view<const T, IdxT>(
            batch_weights.data_handle(), current_batch_size);

          auto L2NormBatch_view =
            raft::make_device_vector_view<T, IdxT>(L2NormBatch.data_handle(), current_batch_size);

          if (metric == cuvs::distance::DistanceType::L2Expanded ||
              metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
            raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
              dev_res,
              raft::make_device_matrix_view<const T, IdxT>(
                data_batch.data(), current_batch_size, n_features),
              L2NormBatch_view);
          }

          auto L2NormBatch_const = raft::make_const_mdspan(L2NormBatch_view);

          auto minClusterAndDistance_view =
            raft::make_device_vector_view<raft::KeyValuePair<IdxT, T>, IdxT>(
              minClusterAndDistance.data_handle(), current_batch_size);

          cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<T, IdxT>(
            dev_res,
            batch_data_view,
            rank_centroids_const,
            minClusterAndDistance_view,
            L2NormBatch_const,
            L2NormBuf_OR_DistBuf,
            metric,
            params.batch_samples,
            params.batch_centroids,
            workspace);

          auto minClusterAndDistance_const = raft::make_const_mdspan(minClusterAndDistance_view);

          cuvs::cluster::kmeans::detail::accumulate_batch_centroids<T, IdxT>(
            dev_res,
            batch_data_view,
            minClusterAndDistance_const,
            batch_weights_view,
            centroid_sums.view(),
            weight_per_cluster.view());

          raft::linalg::map(
            dev_res,
            minClusterAndDistance_view,
            [=] __device__(const raft::KeyValuePair<IdxT, T> kvp, T wt) {
              raft::KeyValuePair<IdxT, T> res;
              res.value = kvp.value * wt;
              res.key   = kvp.key;
              return res;
            },
            raft::make_const_mdspan(minClusterAndDistance_view),
            batch_weights_view);

          cuvs::cluster::kmeans::detail::computeClusterCost(
            dev_res,
            minClusterAndDistance_view,
            workspace,
            raft::make_device_scalar_view(batch_clustering_cost.data_handle()),
            raft::value_op{},
            raft::add_op{});

          raft::linalg::add(dev_res,
                            raft::make_const_mdspan(clustering_cost.view()),
                            raft::make_const_mdspan(batch_clustering_cost.view()),
                            clustering_cost.view());
        }
      }

      // Phase 2: grouped allreduce
      SNMG_GROUP_START();
      SNMG_ALLREDUCE(
        centroid_sums.data_handle(), centroid_sums.data_handle(), n_clusters * n_features);
      SNMG_ALLREDUCE(
        weight_per_cluster.data_handle(), weight_per_cluster.data_handle(), n_clusters);
      SNMG_ALLREDUCE(clustering_cost.data_handle(), clustering_cost.data_handle(), 1);
      SNMG_GROUP_END();
      raft::resource::sync_stream(dev_res);

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

      // Phase 4: convergence check — synchronized across all ranks
      cuvs::cluster::kmeans::detail::compute_centroid_shift<T, IdxT>(
        dev_res,
        raft::make_const_mdspan(rank_centroids.view()),
        raft::make_const_mdspan(new_centroids.view()),
        sqrd_norm_error_dev.view());
      T sqrdNormError = T{0};
      raft::copy(&sqrdNormError, sqrd_norm_error_dev.data_handle(), 1, stream);
      raft::resource::sync_stream(dev_res);

      raft::copy(
        rank_centroids.data_handle(), new_centroids.data_handle(), n_clusters * n_features, stream);

      bool done = false;

      raft::copy(&local_inertia, clustering_cost.data_handle(), 1, stream);
      raft::resource::sync_stream(dev_res);

      if (local_inertia == T{0}) {
        RAFT_LOG_WARN("Zero clustering cost detected: all points coincide with their centroids.");
      } else if (local_n_iter > 1 && prior_cluster_cost > T{0}) {
        T delta = local_inertia / prior_cluster_cost;
        if (delta > 1 - params.tol) { done = true; }
      }
      prior_cluster_cost = local_inertia;

      if (sqrdNormError < params.tol) { done = true; }

      // Allreduce the convergence flag so all ranks agree (prevents NCCL deadlock
      // from floating-point non-determinism in compute_centroid_shift)
      int64_t done_val = done ? 1 : 0;
      raft::copy(d_done.data_handle(), &done_val, 1, stream);
      raft::resource::sync_stream(dev_res);
      SNMG_ALLREDUCE(d_done.data_handle(), d_done.data_handle(), 1);
      raft::resource::sync_stream(dev_res);
      raft::copy(&done_val, d_done.data_handle(), 1, stream);
      raft::resource::sync_stream(dev_res);
      done = (done_val > 0);

      if (done) {
        RAFT_LOG_DEBUG(
          "SNMG KMeans: threshold triggered after %d iterations on rank %d", local_n_iter, rank);
        break;
      }
    }

    // Final inertia recomputation against converged centroids
    raft::matrix::fill(dev_res, clustering_cost.view(), T{0});
    if (has_data) {
      auto rank_centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        rank_centroids.data_handle(), n_clusters, n_features);

      auto& data_batches = *data_batches_opt;
      data_batches.reset();
      for (const auto& data_batch : data_batches) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        cuvs::cluster::kmeans::detail::copy_and_scale_batch_weights<T, IdxT>(
          dev_res,
          sample_weight,
          static_cast<IdxT>(data_batch.offset()),
          current_batch_size,
          weight_scale,
          batch_weights.view());

        std::optional<raft::device_vector_view<const T, IdxT>> batch_sw = std::nullopt;
        if (sample_weight.has_value()) {
          batch_sw = raft::make_device_vector_view<const T, IdxT>(batch_weights.data_handle(),
                                                                  current_batch_size);
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
      }
    }
    SNMG_ALLREDUCE(clustering_cost.data_handle(), clustering_cost.data_handle(), 1);
    raft::resource::sync_stream(dev_res);
    raft::copy(&local_inertia, clustering_cost.data_handle(), 1, stream);
    raft::resource::sync_stream(dev_res);

    RAFT_LOG_DEBUG("SNMG KMeans: n_init %d/%d completed, inertia=%f, n_iter=%d on rank %d",
                   seed_iter + 1,
                   n_init,
                   static_cast<double>(local_inertia),
                   local_n_iter,
                   rank);

    // Best-of-n_init tracking
    if (n_init > 1 && local_inertia < best_inertia) {
      best_inertia = local_inertia;
      best_n_iter  = local_n_iter;
      raft::copy(best_centroids.data_handle(),
                 rank_centroids.data_handle(),
                 n_clusters * n_features,
                 stream);
    }
  }

  // --- Final output (rank 0 writes to caller-provided views) ---
  if (n_init > 1) {
    raft::copy(
      rank_centroids.data_handle(), best_centroids.data_handle(), n_clusters * n_features, stream);
    local_inertia = best_inertia;
    local_n_iter  = best_n_iter;
  }

  if (rank == 0) {
    raft::copy(
      centroids.data_handle(), rank_centroids.data_handle(), n_clusters * n_features, stream);
    raft::resource::sync_stream(dev_res);
    inertia[0] = local_inertia;
    n_iter[0]  = local_n_iter;
  }
}

// ---------------------------------------------------------------------------
// batched_fit_omp — OpenMP wrapper for Path 1 (cuVS / SNMG)
// ---------------------------------------------------------------------------
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
    if (sample_weight.has_value()) {
      sw_local =
        raft::make_host_vector_view<const T, IdxT>(sample_weight->data_handle() + offset, n_local);
    }

    mnmg_fit<T, IdxT>(clique, params, X_local, sw_local, centroids, inertia, n_iter);
  }
}

// Undef local macros
#undef SNMG_ALLREDUCE
#undef SNMG_BCAST
#undef SNMG_GROUP_START
#undef SNMG_GROUP_END

}  // namespace cuvs::cluster::kmeans::mg::detail
