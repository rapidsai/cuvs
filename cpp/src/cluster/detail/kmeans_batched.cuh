/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"
#include "kmeans_common.cuh"

#include "../../neighbors/detail/ann_utils.cuh"
#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Initialize centroids from host data
 *
 * @tparam T      Input data type
 * @tparam IdxT   Index type
 */
template <typename T, typename IdxT>
void init_centroids_from_host_sample(raft::resources const& handle,
                                     const cuvs::cluster::kmeans::params& params,
                                     IdxT batch_size,
                                     raft::host_matrix_view<const T, IdxT> X,
                                     raft::device_matrix_view<T, IdxT> centroids,
                                     rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    IdxT init_sample_size = 3 * batch_size;
    if (init_sample_size < n_clusters) { init_sample_size = 3 * n_clusters; }
    init_sample_size = std::min(init_sample_size, n_samples);

    auto init_sample = raft::make_device_matrix<T, IdxT>(handle, init_sample_size, n_features);
    raft::random::RngState random_state(params.rng_state.seed);
    raft::matrix::sample_rows(handle, random_state, X, init_sample.view());

    auto init_sample_view = raft::make_device_matrix_view<const T, IdxT>(
      init_sample.data_handle(), init_sample_size, n_features);

    if (params.oversampling_factor == 0) {
      cuvs::cluster::kmeans::detail::kmeansPlusPlus<T, IdxT>(
        handle, params, init_sample_view, centroids, workspace);
    } else {
      cuvs::cluster::kmeans::detail::initScalableKMeansPlusPlus<T, IdxT>(
        handle, params, init_sample_view, centroids, workspace);
    }
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    raft::random::RngState random_state(params.rng_state.seed);
    raft::matrix::sample_rows(handle, random_state, X, centroids);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
    // already provided
  } else {
    RAFT_FAIL("Unknown initialization method");
  }
}

/**
 * @brief Accumulate partial centroid sums and counts from a batch
 *
 * This function adds the partial sums from a batch to the running accumulators.
 * It does NOT divide - that happens once at the end of all batches.
 */
template <typename MathT, typename IdxT>
void accumulate_batch_centroids(
  raft::resources const& handle,
  raft::device_matrix_view<const MathT, IdxT> batch_data,
  raft::device_vector_view<const raft::KeyValuePair<IdxT, MathT>, IdxT> minClusterAndDistance,
  raft::device_vector_view<const MathT, IdxT> sample_weights,
  raft::device_matrix_view<MathT, IdxT> centroid_sums,
  raft::device_vector_view<MathT, IdxT> cluster_counts)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_features     = batch_data.extent(1);
  auto n_clusters     = centroid_sums.extent(0);

  auto workspace = rmm::device_uvector<char>(
    batch_data.extent(0), stream, raft::resource::get_workspace_resource(handle));

  auto batch_sums   = raft::make_device_matrix<MathT, IdxT>(handle, n_clusters, n_features);
  auto batch_counts = raft::make_device_vector<MathT, IdxT>(handle, n_clusters);

  raft::matrix::fill(handle, batch_sums.view(), MathT{0});
  raft::matrix::fill(handle, batch_counts.view(), MathT{0});

  cuvs::cluster::kmeans::detail::KeyValueIndexOp<IdxT, MathT> conversion_op;
  thrust::transform_iterator<cuvs::cluster::kmeans::detail::KeyValueIndexOp<IdxT, MathT>,
                             const raft::KeyValuePair<IdxT, MathT>*>
    labels_itr(minClusterAndDistance.data_handle(), conversion_op);

  cuvs::cluster::kmeans::detail::compute_centroid_adjustments(handle,
                                                              batch_data,
                                                              sample_weights,
                                                              labels_itr,
                                                              static_cast<IdxT>(n_clusters),
                                                              batch_sums.view(),
                                                              batch_counts.view(),
                                                              workspace);

  raft::linalg::add(centroid_sums.data_handle(),
                    centroid_sums.data_handle(),
                    batch_sums.data_handle(),
                    centroid_sums.size(),
                    stream);

  raft::linalg::add(cluster_counts.data_handle(),
                    cluster_counts.data_handle(),
                    batch_counts.data_handle(),
                    cluster_counts.size(),
                    stream);
}

/**
 * @brief Main fit function for batched k-means with host data (full-batch / Lloyd's algorithm).
 *
 * Processes host data in GPU-sized batches per iteration, accumulating partial centroid
 * sums and counts, then finalizes centroids at the end of each iteration.
 *
 * @tparam T         Input data type (float, double)
 * @tparam IdxT      Index type (int, int64_t)
 *
 * @param[in]     handle        RAFT resources handle
 * @param[in]     params        K-means parameters
 * @param[in]     X             Input data on HOST [n_samples x n_features]
 * @param[in]     sample_weight Optional weights per sample (on host)
 * @param[inout]  centroids     Initial/output cluster centers [n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances to nearest centroid
 * @param[out]    n_iter        Number of iterations run
 */
template <typename T, typename IdxT>
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const T, IdxT> X,
         std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
         raft::device_matrix_view<T, IdxT> centroids,
         raft::host_scalar_view<T> inertia,
         raft::host_scalar_view<IdxT> n_iter)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  // Read batch_size from params; default to n_samples if 0 (auto)
  IdxT batch_size = static_cast<IdxT>(params.batch_size);
  if (batch_size <= 0) { batch_size = static_cast<IdxT>(n_samples); }

  RAFT_EXPECTS(batch_size > 0, "batch_size must be positive");

  // Warn if user explicitly set batch_size larger than dataset size
  if (params.batch_size > 0 && static_cast<IdxT>(params.batch_size) > n_samples) {
    RAFT_LOG_WARN(
      "batch_size (%zu) is larger than dataset size (%zu). "
      "batch_size will be effectively clamped to %zu.",
      static_cast<size_t>(params.batch_size),
      static_cast<size_t>(n_samples),
      static_cast<size_t>(n_samples));
  }

  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IdxT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");

  RAFT_LOG_DEBUG("KMeans batched fit: n_samples=%zu, n_features=%zu, n_clusters=%d, batch_size=%zu",
                 static_cast<size_t>(n_samples),
                 static_cast<size_t>(n_features),
                 n_clusters,
                 static_cast<size_t>(batch_size));

  rmm::device_uvector<char> workspace(0, stream);

  auto n_init = params.n_init;
  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array && n_init != 1) {
    RAFT_LOG_DEBUG(
      "Explicit initial center position passed: performing only one init in "
      "k-means instead of n_init=%d",
      n_init);
    n_init = 1;
  }

  auto best_centroids = n_init > 1
                          ? raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features)
                          : raft::make_device_matrix<T, IdxT>(handle, 0, 0);
  T best_inertia      = std::numeric_limits<T>::max();
  IdxT best_n_iter    = 0;

  std::mt19937 gen(params.rng_state.seed);

  // ----- Allocate reusable work buffers (shared across n_init iterations) -----
  auto batch_data    = raft::make_device_matrix<T, IdxT>(handle, batch_size, n_features);
  auto batch_weights = raft::make_device_vector<T, IdxT>(handle, batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IdxT, T>, IdxT>(handle, batch_size);
  auto L2NormBatch = raft::make_device_vector<T, IdxT>(handle, batch_size);
  rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);

  auto centroid_sums  = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);
  auto cluster_counts = raft::make_device_vector<T, IdxT>(handle, n_clusters);
  auto new_centroids  = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);

  // ---- Main n_init loop ----
  for (int seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    cuvs::cluster::kmeans::params iter_params = params;
    iter_params.rng_state.seed                = gen();

    RAFT_LOG_DEBUG("KMeans batched fit: n_init iteration %d/%d (seed=%llu)",
                   seed_iter + 1,
                   n_init,
                   (unsigned long long)iter_params.rng_state.seed);

    if (iter_params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
      init_centroids_from_host_sample(handle, iter_params, batch_size, X, centroids, workspace);
    }

    if (!sample_weight.has_value()) { raft::matrix::fill(handle, batch_weights.view(), T{1}); }

    // Reset per-iteration state
    T prior_cluster_cost = 0;

    for (n_iter[0] = 1; n_iter[0] <= iter_params.max_iter; ++n_iter[0]) {
      RAFT_LOG_DEBUG("KMeans batched: Iteration %d", n_iter[0]);

      raft::copy(new_centroids.data_handle(), centroids.data_handle(), centroids.size(), stream);

      raft::matrix::fill(handle, centroid_sums.view(), T{0});
      raft::matrix::fill(handle, cluster_counts.view(), T{0});
      auto clustering_cost = raft::make_device_scalar<T>(handle, T{0});

      auto centroids_const = raft::make_const_mdspan(centroids);

      using namespace cuvs::spatial::knn::detail::utils;
      batch_load_iterator<T> data_batches(
        X.data_handle(), n_samples, n_features, batch_size, stream);

      for (const auto& data_batch : data_batches) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        if (sample_weight.has_value()) {
          raft::copy(batch_weights.data_handle(),
                     sample_weight->data_handle() + data_batch.offset(),
                     current_batch_size,
                     stream);
        }

        auto batch_weights_view = raft::make_device_vector_view<const T, IdxT>(
          batch_weights.data_handle(), current_batch_size);

        if (metric == cuvs::distance::DistanceType::L2Expanded ||
            metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
            L2NormBatch.data_handle(), data_batch.data(), n_features, current_batch_size, stream);
        }

        auto L2NormBatch_const = raft::make_device_vector_view<const T, IdxT>(
          L2NormBatch.data_handle(), current_batch_size);

        cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<T, IdxT>(
          handle,
          batch_data_view,
          centroids_const,
          minClusterAndDistance.view(),
          L2NormBatch_const,
          L2NormBuf_OR_DistBuf,
          metric,
          params.batch_samples,
          params.batch_centroids,
          workspace);

        auto minClusterAndDistance_const = raft::make_const_mdspan(minClusterAndDistance.view());

        accumulate_batch_centroids<T, IdxT>(handle,
                                            batch_data_view,
                                            minClusterAndDistance_const,
                                            batch_weights_view,
                                            centroid_sums.view(),
                                            cluster_counts.view());

        if (params.inertia_check || n_iter[0] == iter_params.max_iter) {
          // Compute cluster cost for this batch and accumulate
          cuvs::cluster::kmeans::detail::computeClusterCost(handle,
                                                            minClusterAndDistance.view(),
                                                            workspace,
                                                            clustering_cost.view(),
                                                            raft::value_op{},
                                                            raft::add_op{});
        }
      }

      auto centroid_sums_const = raft::make_device_matrix_view<const T, IdxT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto cluster_counts_const =
        raft::make_device_vector_view<const T, IdxT>(cluster_counts.data_handle(), n_clusters);

      finalize_centroids<T, IdxT>(
        handle, centroid_sums_const, cluster_counts_const, centroids_const, new_centroids.view());

      // Convergence check
      T sqrdNormError =
        compute_centroid_shift<T, IdxT>(handle,
                                        raft::make_device_matrix_view<const T, IdxT>(
                                          centroids.data_handle(), n_clusters, n_features),
                                        raft::make_device_matrix_view<const T, IdxT>(
                                          new_centroids.data_handle(), n_clusters, n_features));

      raft::copy(centroids.data_handle(), new_centroids.data_handle(), centroids.size(), stream);

      bool done = false;
      if (params.inertia_check) {
        raft::copy(inertia.data_handle(), clustering_cost.data_handle(), 1, stream);
        raft::resource::sync_stream(handle);
        if (n_iter[0] > 1) {
          T delta = inertia[0] / prior_cluster_cost;
          if (delta > 1 - params.tol) done = true;
        }
        prior_cluster_cost = inertia[0];
      }

      if (sqrdNormError < params.tol) done = true;

      if (done || n_iter[0] == iter_params.max_iter) {
        RAFT_LOG_DEBUG("KMeans batched: Converged after %d iterations", n_iter[0]);
        // Inertia for the last iteration is always computed
        if (!params.inertia_check) {
          raft::copy(inertia.data_handle(), clustering_cost.data_handle(), 1, stream);
          raft::resource::sync_stream(handle);
        }
        break;
      }
    }

    {
      RAFT_LOG_DEBUG("KMeans batched: n_init %d/%d completed with inertia=%f",
                     seed_iter + 1,
                     n_init,
                     static_cast<double>(inertia[0]));

      if (n_init > 1 && inertia[0] < best_inertia) {
        best_inertia = inertia[0];
        best_n_iter  = n_iter[0];
        raft::copy(best_centroids.data_handle(), centroids.data_handle(), centroids.size(), stream);
      }
    }
  }
  if (n_init > 1) {
    raft::copy(centroids.data_handle(), best_centroids.data_handle(), centroids.size(), stream);
    inertia[0] = best_inertia;
    n_iter[0]  = best_n_iter;
    RAFT_LOG_DEBUG("KMeans batched: Best of %d runs: inertia=%f, n_iter=%d",
                   n_init,
                   static_cast<double>(best_inertia),
                   best_n_iter);
  }
}

/**
 * @brief Predict cluster labels for host data using batched processing.
 *
 * @tparam T         Input data type (float, double)
 * @tparam IdxT      Index type (int, int64_t)
 */
template <typename T, typename IdxT>
void predict(raft::resources const& handle,
             const cuvs::cluster::kmeans::params& params,
             raft::host_matrix_view<const T, IdxT> X,
             std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
             raft::device_matrix_view<const T, IdxT> centroids,
             raft::host_vector_view<IdxT, IdxT> labels,
             bool normalize_weight,
             raft::host_scalar_view<T> inertia)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;

  // Read batch_size from params; default to n_samples if 0 (auto)
  IdxT batch_size = static_cast<IdxT>(params.batch_size);
  if (batch_size <= 0) { batch_size = static_cast<IdxT>(n_samples); }

  RAFT_EXPECTS(batch_size > 0, "batch_size must be positive");

  // Warn if user explicitly set batch_size larger than dataset size
  if (params.batch_size > 0 && static_cast<IdxT>(params.batch_size) > n_samples) {
    RAFT_LOG_WARN(
      "batch_size (%zu) is larger than dataset size (%zu). "
      "batch_size will be effectively clamped to %zu.",
      static_cast<size_t>(params.batch_size),
      static_cast<size_t>(n_samples),
      static_cast<size_t>(n_samples));
  }

  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(centroids.extent(0) == static_cast<IdxT>(n_clusters),
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");
  RAFT_EXPECTS(labels.extent(0) == n_samples, "labels.extent(0) must equal n_samples");

  auto batch_data    = raft::make_device_matrix<T, IdxT>(handle, batch_size, n_features);
  auto batch_weights = raft::make_device_vector<T, IdxT>(handle, batch_size);
  auto batch_labels  = raft::make_device_vector<IdxT, IdxT>(handle, batch_size);

  inertia[0] = 0;

  for (IdxT batch_idx = 0; batch_idx < n_samples; batch_idx += batch_size) {
    IdxT current_batch_size = std::min(batch_size, n_samples - batch_idx);

    raft::copy(batch_data.data_handle(),
               X.data_handle() + batch_idx * n_features,
               current_batch_size * n_features,
               stream);

    if (sample_weight.has_value()) {
      raft::copy(batch_weights.data_handle(),
                 sample_weight->data_handle() + batch_idx,
                 current_batch_size,
                 stream);
    }

    auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
      batch_data.data_handle(), current_batch_size, n_features);

    std::optional<raft::device_vector_view<const T, IdxT>> batch_weights_view = std::nullopt;
    if (sample_weight.has_value()) {
      batch_weights_view = std::make_optional(raft::make_device_vector_view<const T, IdxT>(
        batch_weights.data_handle(), current_batch_size));
    }

    auto batch_labels_view =
      raft::make_device_vector_view<IdxT, IdxT>(batch_labels.data_handle(), current_batch_size);

    T batch_inertia = 0;
    cuvs::cluster::kmeans::detail::kmeans_predict<T, IdxT>(
      handle,
      params,
      batch_data_view,
      batch_weights_view,
      centroids,
      batch_labels_view,
      normalize_weight,
      raft::make_host_scalar_view(&batch_inertia));

    raft::copy(
      labels.data_handle() + batch_idx, batch_labels.data_handle(), current_batch_size, stream);

    inertia[0] += batch_inertia;
  }

  raft::resource::sync_stream(handle, stream);
}

/**
 * @brief Fit k-means and predict cluster labels using batched processing.
 */
template <typename T, typename IdxT>
void fit_predict(raft::resources const& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::host_matrix_view<const T, IdxT> X,
                 std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
                 raft::device_matrix_view<T, IdxT> centroids,
                 raft::host_vector_view<IdxT, IdxT> labels,
                 raft::host_scalar_view<T> inertia,
                 raft::host_scalar_view<IdxT> n_iter)
{
  T fit_inertia = 0;
  fit<T, IdxT>(
    handle, params, X, sample_weight, centroids, raft::make_host_scalar_view(&fit_inertia), n_iter);

  auto centroids_const = raft::make_const_mdspan(centroids);

  predict<T, IdxT>(handle, params, X, sample_weight, centroids_const, labels, false, inertia);
}

}  // namespace cuvs::cluster::kmeans::detail
