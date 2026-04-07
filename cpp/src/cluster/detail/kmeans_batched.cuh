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
                                     IdxT streaming_batch_size,
                                     raft::host_matrix_view<const T, IdxT> X,
                                     raft::device_matrix_view<T, IdxT> centroids,
                                     rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    // this is a heuristic to speed up the initialization
    IdxT init_sample_size = 3 * streaming_batch_size;
    if (init_sample_size < n_clusters) { init_sample_size = 3 * n_clusters; }
    init_sample_size = std::min(init_sample_size, n_samples);

    auto init_sample = raft::make_device_matrix<T, IdxT>(handle, init_sample_size, n_features);
    raft::random::RngState random_state(params.rng_state.seed);
    raft::matrix::sample_rows(handle, random_state, X, init_sample.view());

    if (params.oversampling_factor == 0) {
      cuvs::cluster::kmeans::detail::kmeansPlusPlus<T, IdxT>(
        handle, params, raft::make_const_mdspan(init_sample.view()), centroids, workspace);
    } else {
      cuvs::cluster::kmeans::detail::initScalableKMeansPlusPlus<T, IdxT>(
        handle, params, raft::make_const_mdspan(init_sample.view()), centroids, workspace);
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
 * @brief Compute the weight normalization scale factor for host sample weights. Weights are
 * normalized to sum to n_samples. Returns the scale factor to apply to each weight.
 *
 * @param[in] sample_weight Optional host vector of sample weights
 * @param[in] n_samples     Number of samples
 * @return Scale factor (1.0 if no weights or already normalized)
 */
template <typename T, typename IdxT>
T compute_host_weight_scale(
  const std::optional<raft::host_vector_view<const T, IdxT>>& sample_weight, IdxT n_samples)
{
  if (!sample_weight.has_value()) { return T{1}; }

  T wt_sum        = T{0};
  const T* sw_ptr = sample_weight->data_handle();
  for (IdxT i = 0; i < n_samples; ++i) {
    wt_sum += sw_ptr[i];
  }
  if (wt_sum == static_cast<T>(n_samples)) { return T{1}; }

  RAFT_LOG_DEBUG(
    "[Warning!] KMeans: normalizing the user provided sample weight to "
    "sum up to %zu samples (scale=%f)",
    static_cast<size_t>(n_samples),
    static_cast<double>(static_cast<T>(n_samples) / wt_sum));
  return static_cast<T>(n_samples) / wt_sum;
}

/**
 * @brief Copy host sample weights to device and apply normalization scale.
 *
 * When sample_weight is provided, copies the batch slice to the device buffer
 * and applies the normalization scale factor. When not provided, the device
 * buffer is assumed to already be filled with 1.0.
 *
 * @param[in]     handle         RAFT resources handle
 * @param[in]     sample_weight  Optional host weights
 * @param[in]     batch_offset   Offset into the host weights for this batch
 * @param[in]     batch_size     Number of elements in this batch
 * @param[in]     weight_scale   Scale factor from compute_host_weight_scale
 * @param[inout]  batch_weights  Device buffer to write normalized weights into
 */
template <typename T, typename IdxT>
void copy_and_scale_batch_weights(
  raft::resources const& handle,
  const std::optional<raft::host_vector_view<const T, IdxT>>& sample_weight,
  size_t batch_offset,
  IdxT batch_size,
  T weight_scale,
  raft::device_vector<T, IdxT>& batch_weights)
{
  if (!sample_weight.has_value()) { return; }

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::copy(
    batch_weights.data_handle(), sample_weight->data_handle() + batch_offset, batch_size, stream);

  if (weight_scale != T{1}) {
    auto batch_weights_view =
      raft::make_device_vector_view<T, IdxT>(batch_weights.data_handle(), batch_size);
    raft::linalg::map(handle,
                      batch_weights_view,
                      raft::mul_const_op<T>{weight_scale},
                      raft::make_const_mdspan(batch_weights_view));
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
  raft::device_vector_view<MathT, IdxT> cluster_counts,
  raft::device_matrix_view<MathT, IdxT> batch_sums,
  raft::device_vector_view<MathT, IdxT> batch_counts)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto workspace = rmm::device_uvector<char>(
    batch_data.extent(0), stream, raft::resource::get_workspace_resource(handle));

  cuvs::cluster::kmeans::detail::KeyValueIndexOp<IdxT, MathT> conversion_op;
  thrust::transform_iterator<cuvs::cluster::kmeans::detail::KeyValueIndexOp<IdxT, MathT>,
                             const raft::KeyValuePair<IdxT, MathT>*>
    labels_itr(minClusterAndDistance.data_handle(), conversion_op);

  cuvs::cluster::kmeans::detail::compute_centroid_adjustments(
    handle,
    batch_data,
    sample_weights,
    labels_itr,
    static_cast<IdxT>(centroid_sums.extent(0)),
    batch_sums,
    batch_counts,
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
         raft::host_scalar_view<int> n_iter)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  IdxT streaming_batch_size = static_cast<IdxT>(params.streaming_batch_size);

  if (params.streaming_batch_size == 0) {
    streaming_batch_size = static_cast<IdxT>(n_samples);
  } else if (params.streaming_batch_size < 0 || params.streaming_batch_size > n_samples) {
    RAFT_LOG_WARN("streaming_batch_size must be >= 0 and <= n_samples, using n_samples=%zu",
                  static_cast<size_t>(n_samples));
    streaming_batch_size = static_cast<IdxT>(n_samples);
  }

  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IdxT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");

  RAFT_LOG_DEBUG(
    "KMeans batched fit: n_samples=%zu, n_features=%zu, n_clusters=%d, streaming_batch_size=%zu",
    static_cast<size_t>(n_samples),
    static_cast<size_t>(n_features),
    n_clusters,
    static_cast<size_t>(streaming_batch_size));

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
  int best_n_iter    = 0;

  std::mt19937 gen(params.rng_state.seed);

  // ----- Allocate reusable work buffers (shared across n_init iterations) -----
  auto batch_weights = raft::make_device_vector<T, IdxT>(handle, streaming_batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IdxT, T>, IdxT>(handle, streaming_batch_size);
  auto L2NormBatch = raft::make_device_vector<T, IdxT>(handle, streaming_batch_size);
  rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);

  auto centroid_sums         = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);
  auto weight_per_cluster    = raft::make_device_vector<T, IdxT>(handle, n_clusters);
  auto new_centroids         = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);
  auto clustering_cost       = raft::make_device_vector<T, IdxT>(handle, 1);
  auto batch_clustering_cost = raft::make_device_vector<T, IdxT>(handle, 1);
  auto batch_sums            = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);
  auto batch_counts          = raft::make_device_vector<T, IdxT>(handle, n_clusters);

  // Compute weight normalization (matches checkWeight in regular kmeans)
  T weight_scale = compute_host_weight_scale(sample_weight, n_samples);

  // ---- Main n_init loop ----
  for (int seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    cuvs::cluster::kmeans::params iter_params = params;
    iter_params.rng_state.seed                = gen();

    RAFT_LOG_DEBUG("KMeans batched fit: n_init iteration %d/%d (seed=%llu)",
                   seed_iter + 1,
                   n_init,
                   (unsigned long long)iter_params.rng_state.seed);

    if (iter_params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
      init_centroids_from_host_sample(
        handle, iter_params, streaming_batch_size, X, centroids, workspace);
    }

    if (!sample_weight.has_value()) { raft::matrix::fill(handle, batch_weights.view(), T{1}); }

    // Reset per-iteration state
    T prior_cluster_cost = 0;

    cuvs::spatial::knn::detail::utils::batch_load_iterator<T> data_batches(
      X.data_handle(), n_samples, n_features, streaming_batch_size, stream);

    for (n_iter[0] = 1; n_iter[0] <= iter_params.max_iter; ++n_iter[0]) {
      RAFT_LOG_DEBUG("KMeans batched: Iteration %d", n_iter[0]);

      raft::matrix::fill(handle, centroid_sums.view(), T{0});
      raft::matrix::fill(handle, weight_per_cluster.view(), T{0});

      raft::matrix::fill(handle, clustering_cost.view(), T{0});

      auto centroids_const = raft::make_const_mdspan(centroids);

      for (const auto& data_batch : data_batches) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());
        raft::matrix::fill(handle, batch_clustering_cost.view(), T{0});

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        copy_and_scale_batch_weights(handle,
                                     sample_weight,
                                     data_batch.offset(),
                                     current_batch_size,
                                     weight_scale,
                                     batch_weights);

        auto batch_weights_view = raft::make_device_vector_view<const T, IdxT>(
          batch_weights.data_handle(), current_batch_size);

        auto L2NormBatch_view =
          raft::make_device_vector_view<T, IdxT>(L2NormBatch.data_handle(), current_batch_size);

        if (metric == cuvs::distance::DistanceType::L2Expanded ||
            metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
            handle,
            raft::make_device_matrix_view<const T, IdxT>(
              data_batch.data(), current_batch_size, n_features),
            L2NormBatch_view);
        }

        auto L2NormBatch_const = raft::make_const_mdspan(L2NormBatch_view);

        auto minClusterAndDistance_view =
          raft::make_device_vector_view<raft::KeyValuePair<IdxT, T>, IdxT>(
            minClusterAndDistance.data_handle(), current_batch_size);

        cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<T, IdxT>(
          handle,
          batch_data_view,
          centroids_const,
          minClusterAndDistance_view,
          L2NormBatch_const,
          L2NormBuf_OR_DistBuf,
          metric,
          params.batch_samples,
          params.batch_centroids,
          workspace);

        auto minClusterAndDistance_const = raft::make_const_mdspan(minClusterAndDistance_view);

        accumulate_batch_centroids<T, IdxT>(handle,
                                            batch_data_view,
                                            minClusterAndDistance_const,
                                            batch_weights_view,
                                            centroid_sums.view(),
                                            weight_per_cluster.view(),
                                            batch_sums.view(),
                                            batch_counts.view());

        if (params.inertia_check) {
          raft::linalg::map(
            handle,
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
            handle,
            minClusterAndDistance_view,
            workspace,
            raft::make_device_scalar_view(batch_clustering_cost.data_handle()),
            raft::value_op{},
            raft::add_op{});
          raft::linalg::add(handle,
                            raft::make_const_mdspan(clustering_cost.view()),
                            raft::make_const_mdspan(batch_clustering_cost.view()),
                            clustering_cost.view());
        }
      }

      auto centroid_sums_const = raft::make_device_matrix_view<const T, IdxT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto weight_per_cluster_const =
        raft::make_device_vector_view<const T, IdxT>(weight_per_cluster.data_handle(), n_clusters);

      finalize_centroids<T, IdxT>(handle,
                                  centroid_sums_const,
                                  weight_per_cluster_const,
                                  centroids_const,
                                  new_centroids.view());

      T sqrdNormError = compute_centroid_shift<T, IdxT>(
        handle, raft::make_const_mdspan(centroids), raft::make_const_mdspan(new_centroids.view()));

      raft::copy(handle, centroids, new_centroids.view());

      bool done = false;
      if (params.inertia_check) {
        raft::copy(inertia.data_handle(), clustering_cost.data_handle(), 1, stream);
        raft::resource::sync_stream(handle);
        ASSERT(inertia[0] != (T)0.0,
               "Too few points and centroids being found is getting 0 cost from "
               "centers");
        if (n_iter[0] > 1) {
          T delta = inertia[0] / prior_cluster_cost;
          if (delta > 1 - params.tol) done = true;
        }
        prior_cluster_cost = inertia[0];
      }

      if (sqrdNormError < params.tol) done = true;

      if (done) {
        RAFT_LOG_DEBUG("Threshold triggered after %d iterations. Terminating early.", n_iter[0]);
        break;
      }
    }

    // Recompute final weighted inertia with the converged centroids.
    {
      auto centroids_const_view = raft::make_device_matrix_view<const T, IdxT>(
        centroids.data_handle(), n_clusters, n_features);

      inertia[0] = T{0};
      for (const auto& data_batch : data_batches) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        std::optional<raft::device_vector_view<const T, IdxT>> batch_sw = std::nullopt;
        if (sample_weight.has_value()) {
          copy_and_scale_batch_weights(handle,
                                       sample_weight,
                                       data_batch.offset(),
                                       current_batch_size,
                                       weight_scale,
                                       batch_weights);
          batch_sw = raft::make_device_vector_view<const T, IdxT>(batch_weights.data_handle(),
                                                                  current_batch_size);
        }

        T batch_cost = T{0};
        cuvs::cluster::kmeans::cluster_cost(handle,
                                            batch_data_view,
                                            centroids_const_view,
                                            raft::make_host_scalar_view(&batch_cost),
                                            batch_sw);

        inertia[0] += batch_cost;
      }
    }

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
  if (n_init > 1) {
    inertia[0] = best_inertia;
    n_iter[0]  = best_n_iter;
    raft::copy(handle, centroids, best_centroids.view());
    RAFT_LOG_DEBUG("KMeans batched: Best of %d runs: inertia=%f, n_iter=%d",
                   n_init,
                   static_cast<double>(best_inertia),
                   best_n_iter);
  }
}

}  // namespace cuvs::cluster::kmeans::detail
