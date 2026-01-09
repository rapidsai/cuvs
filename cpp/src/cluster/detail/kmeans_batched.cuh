/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans_common.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace cuvs::cluster::kmeans::batched::detail {

/**
 * @brief Sample data from host to device for initialization
 *
 * Samples `n_samples_to_gather` rows from host data and copies to device.
 * Uses uniform strided sampling for simplicity and cache-friendliness.
 */
template <typename DataT, typename IndexT>
void prepare_init_sample(raft::resources const& handle,
                           raft::host_matrix_view<const DataT, IndexT> X,
                           raft::device_matrix_view<DataT, IndexT> X_sample,
                           uint64_t seed)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_samples       = X.extent(0);
  auto n_features      = X.extent(1);
  auto n_samples_out   = X_sample.extent(0);

  // Use strided sampling for cache-friendliness
  // For truly random, could use std::shuffle on indices first
  std::mt19937 gen(seed);
  std::vector<IndexT> indices(n_samples);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  std::vector<DataT> host_sample(n_samples_out * n_features);

#pragma omp parallel for
  for (IndexT i = 0; i < static_cast<IndexT>(n_samples_out); i++) {
    IndexT src_idx = indices[i];
    std::memcpy(host_sample.data() + i * n_features,
                X.data_handle() + src_idx * n_features,
                n_features * sizeof(DataT));
  }

  raft::copy(X_sample.data_handle(), host_sample.data(), host_sample.size(), stream);
}

/**
 * @brief Initialize centroids using k-means++ on a sample of the host data
 */
template <typename DataT, typename IndexT>
void init_centroids_from_host_sample(raft::resources const& handle,
                                     const cuvs::cluster::kmeans::params& params,
                                     raft::host_matrix_view<const DataT, IndexT> X,
                                     raft::device_matrix_view<DataT, IndexT> centroids,
                                     rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;

  // Sample size for initialization: at least 3 * n_clusters, but not more than n_samples
  size_t init_sample_size = std::min(static_cast<size_t>(n_samples),
                                     std::max(static_cast<size_t>(3 * n_clusters),
                                              static_cast<size_t>(10000)));

  RAFT_LOG_DEBUG("KMeans batched: sampling %zu points for initialization", init_sample_size);

  // Sample data from host to device
  auto init_sample = raft::make_device_matrix<DataT, IndexT>(handle, init_sample_size, n_features);
  prepare_init_sample(handle, X, init_sample.view(), params.rng_state.seed);

  // Run k-means++ on the sample
  if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    cuvs::cluster::kmeans::detail::kmeansPlusPlus<DataT, IndexT>(
      handle,
      params,
      raft::make_device_matrix_view<const DataT, IndexT>(
        init_sample.data_handle(), init_sample_size, n_features),
      centroids,
      workspace);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    // Just use the first n_clusters samples
    raft::copy(centroids.data_handle(),
               init_sample.data_handle(),
               n_clusters * n_features,
               stream);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
    // Centroids already provided, nothing to do
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
template <typename DataT, typename IndexT>
void accumulate_batch_centroids(
  raft::resources const& handle,
  raft::device_matrix_view<const DataT, IndexT> batch_data,
  raft::device_vector_view<const raft::KeyValuePair<IndexT, DataT>, IndexT> minClusterAndDistance,
  raft::device_vector_view<const DataT, IndexT> sample_weights,
  raft::device_matrix_view<DataT, IndexT> centroid_sums,
  raft::device_vector_view<DataT, IndexT> cluster_counts,
  rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = batch_data.extent(0);
  auto n_features     = batch_data.extent(1);
  auto n_clusters     = centroid_sums.extent(0);

  // Temporary buffers for this batch's partial results
  auto batch_sums   = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto batch_counts = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);

  // Zero the batch temporaries
  thrust::fill(raft::resource::get_thrust_policy(handle),
               batch_sums.data_handle(),
               batch_sums.data_handle() + batch_sums.size(),
               DataT{0});
  thrust::fill(raft::resource::get_thrust_policy(handle),
               batch_counts.data_handle(),
               batch_counts.data_handle() + batch_counts.size(),
               DataT{0});

  // Extract cluster labels from KeyValuePair
  cuvs::cluster::kmeans::detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
  thrust::transform_iterator<cuvs::cluster::kmeans::detail::KeyValueIndexOp<IndexT, DataT>,
                             const raft::KeyValuePair<IndexT, DataT>*>
    labels_itr(minClusterAndDistance.data_handle(), conversion_op);

  workspace.resize(n_samples, stream);

  // Compute weighted sum of samples per cluster for this batch
  raft::linalg::reduce_rows_by_key(const_cast<DataT*>(batch_data.data_handle()),
                                   batch_data.extent(1),
                                   labels_itr,
                                   sample_weights.data_handle(),
                                   workspace.data(),
                                   batch_data.extent(0),
                                   batch_data.extent(1),
                                   n_clusters,
                                   batch_sums.data_handle(),
                                   stream);

  // Compute sum of weights per cluster for this batch
  raft::linalg::reduce_cols_by_key(sample_weights.data_handle(),
                                   labels_itr,
                                   batch_counts.data_handle(),
                                   static_cast<IndexT>(1),
                                   static_cast<IndexT>(n_samples),
                                   static_cast<IndexT>(n_clusters),
                                   stream);

  // Add batch results to running accumulators
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
 * @brief Finalize centroids by dividing accumulated sums by counts
 */
template <typename DataT, typename IndexT>
void finalize_centroids(raft::resources const& handle,
                        raft::device_matrix_view<const DataT, IndexT> centroid_sums,
                        raft::device_vector_view<const DataT, IndexT> cluster_counts,
                        raft::device_matrix_view<const DataT, IndexT> old_centroids,
                        raft::device_matrix_view<DataT, IndexT> new_centroids)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_clusters     = new_centroids.extent(0);
  auto n_features     = new_centroids.extent(1);

  // Copy sums to new_centroids first
  raft::copy(
    new_centroids.data_handle(), centroid_sums.data_handle(), centroid_sums.size(), stream);

  // Divide by counts: new_centroids[i] = centroid_sums[i] / cluster_counts[i]
  // When count is 0, set to 0 (will be fixed below)
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
    handle,
    raft::make_const_mdspan(new_centroids),
    cluster_counts,
    new_centroids,
    raft::div_checkzero_op{});

  // Copy old centroids to new centroids where cluster_counts[i] == 0
  cub::ArgIndexInputIterator<const DataT*> itr_wt(cluster_counts.data_handle());
  raft::matrix::gather_if(
    old_centroids.data_handle(),
    static_cast<int>(old_centroids.extent(1)),
    static_cast<int>(old_centroids.extent(0)),
    itr_wt,
    itr_wt,
    static_cast<int>(cluster_counts.size()),
    new_centroids.data_handle(),
    [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) {
      return map.value == DataT{0};  // predicate: copy when count is 0
    },
    raft::key_op{},
    stream);
}

/**
 * @brief Main fit function for batched k-means with host data
 *
 * @tparam DataT  Data type (float, double)
 * @tparam IndexT Index type (int, int64_t)
 *
 * @param[in]     handle        RAFT resources handle
 * @param[in]     params        K-means parameters
 * @param[in]     X        Input data on HOST [n_samples x n_features]
 * @param[in]     batch_size    Number of samples to process per batch
 * @param[in]     sample_weight Optional weights per sample (on host)
 * @param[inout]  centroids     Initial/output cluster centers [n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances to nearest centroid
 * @param[out]    n_iter        Number of iterations run
 */
template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const DataT, IndexT> X,
         IndexT batch_size,
         std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  RAFT_EXPECTS(batch_size > 0, "batch_size must be positive");
  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IndexT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features,
               "centroids.extent(1) must equal n_features");

  raft::default_logger().set_level(params.verbosity);

  RAFT_LOG_DEBUG(
    "KMeans batched fit: n_samples=%zu, n_features=%zu, n_clusters=%d, batch_size=%zu",
    static_cast<size_t>(n_samples),
    static_cast<size_t>(n_features),
    n_clusters,
    static_cast<size_t>(batch_size));

  rmm::device_uvector<char> workspace(0, stream);

  // Initialize centroids from a sample of host data
  if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
    init_centroids_from_host_sample(handle, params, X, centroids, workspace);
  }

  // Allocate device buffers
  // Batch buffer for data
  auto batch_data = raft::make_device_matrix<DataT, IndexT>(handle, batch_size, n_features);
  // Batch buffer for weights
  auto batch_weights = raft::make_device_vector<DataT, IndexT>(handle, batch_size);
  // Cluster assignment for batch
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, batch_size);
  // L2 norms of batch data
  auto L2NormBatch = raft::make_device_vector<DataT, IndexT>(handle, batch_size);
  // Temporary buffer for distance computation
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // Accumulators for centroid computation (persist across batches within an iteration)
  auto centroid_sums   = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto cluster_counts  = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);
  auto new_centroids   = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  // Host buffer for batch data (pinned memory for faster H2D transfer)
  std::vector<DataT> host_batch_buffer(batch_size * n_features);
  std::vector<DataT> host_weight_buffer(batch_size);

  // Cluster cost for convergence check
  rmm::device_scalar<DataT> clusterCostD(stream);
  DataT priorClusteringCost = 0;

  // Main iteration loop
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    RAFT_LOG_DEBUG("KMeans batched: Iteration %d", n_iter[0]);

    // Zero accumulators at start of each iteration
    thrust::fill(raft::resource::get_thrust_policy(handle),
                 centroid_sums.data_handle(),
                 centroid_sums.data_handle() + centroid_sums.size(),
                 DataT{0});
    thrust::fill(raft::resource::get_thrust_policy(handle),
                 cluster_counts.data_handle(),
                 cluster_counts.data_handle() + cluster_counts.size(),
                 DataT{0});

    DataT total_cost = 0;

    // Process all data in batches
    for (IndexT offset = 0; offset < n_samples; offset += batch_size) {
      IndexT current_batch_size = std::min(batch_size, n_samples - offset);

      // Copy batch data from host to device
      raft::copy(batch_data.data_handle(),
                 X.data_handle() + offset * n_features,
                 current_batch_size * n_features,
                 stream);

      // Copy or set weights for this batch
      if (sample_weight) {
        raft::copy(batch_weights.data_handle(),
                   sample_weight->data_handle() + offset,
                   current_batch_size,
                   stream);
      } else {
        thrust::fill(raft::resource::get_thrust_policy(handle),
                     batch_weights.data_handle(),
                     batch_weights.data_handle() + current_batch_size,
                     DataT{1});
      }

      // Create views for current batch size
      auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
        batch_data.data_handle(), current_batch_size, n_features);
      auto batch_weights_view = raft::make_device_vector_view<const DataT, IndexT>(
        batch_weights.data_handle(), current_batch_size);
      auto minClusterAndDistance_view =
        raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistance.data_handle(), current_batch_size);
      auto L2NormBatch_view = raft::make_device_vector_view<DataT, IndexT>(
        L2NormBatch.data_handle(), current_batch_size);

      // Compute L2 norms for batch if needed
      if (metric == cuvs::distance::DistanceType::L2Expanded ||
          metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
          L2NormBatch.data_handle(),
          batch_data.data_handle(),
          n_features,
          current_batch_size,
          stream);
      }

      // Find nearest centroid for each sample in batch
      auto centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
        centroids.data_handle(), n_clusters, n_features);
      auto L2NormBatch_const = raft::make_device_vector_view<const DataT, IndexT>(
        L2NormBatch.data_handle(), current_batch_size);

      cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<DataT, IndexT>(
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

      // Accumulate partial sums for this batch
      auto minClusterAndDistance_const =
        raft::make_device_vector_view<const raft::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistance.data_handle(), current_batch_size);

      accumulate_batch_centroids<DataT, IndexT>(handle,
                                                batch_data_view,
                                                minClusterAndDistance_const,
                                                batch_weights_view,
                                                centroid_sums.view(),
                                                cluster_counts.view(),
                                                workspace);

      // Accumulate cluster cost if checking convergence
      if (params.inertia_check) {
        cuvs::cluster::kmeans::detail::computeClusterCost(
          handle,
          minClusterAndDistance_view,
          workspace,
          raft::make_device_scalar_view(clusterCostD.data()),
          raft::value_op{},
          raft::add_op{});
        DataT batch_cost = clusterCostD.value(stream);
        total_cost += batch_cost;
      }
    }  // end batch loop

    // Finalize centroids: divide sums by counts
    auto centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      centroids.data_handle(), n_clusters, n_features);
    auto centroid_sums_const = raft::make_device_matrix_view<const DataT, IndexT>(
      centroid_sums.data_handle(), n_clusters, n_features);
    auto cluster_counts_const = raft::make_device_vector_view<const DataT, IndexT>(
      cluster_counts.data_handle(), n_clusters);

    finalize_centroids<DataT, IndexT>(
      handle, centroid_sums_const, cluster_counts_const, centroids_const, new_centroids.view());

    // Compute squared norm of change in centroids
    auto sqrdNorm = raft::make_device_scalar<DataT>(handle, DataT{0});
    raft::linalg::mapThenSumReduce(sqrdNorm.data_handle(),
                                   new_centroids.size(),
                                   raft::sqdiff_op{},
                                   stream,
                                   centroids.data_handle(),
                                   new_centroids.data_handle());

    DataT sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data_handle(), 1, stream);

    // Update centroids
    raft::copy(centroids.data_handle(), new_centroids.data_handle(), new_centroids.size(), stream);

    // Check convergence
    bool done = false;
    if (params.inertia_check) {
      if (n_iter[0] > 1) {
        DataT delta = total_cost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
      }
      priorClusteringCost = total_cost;
    }

    raft::resource::sync_stream(handle, stream);
    if (sqrdNormError < params.tol) done = true;

    if (done) {
      RAFT_LOG_DEBUG("KMeans batched: Converged after %d iterations", n_iter[0]);
      break;
    }
  }  // end iteration loop

  // Compute final inertia by processing all data once more
  inertia[0] = 0;
  for (IndexT offset = 0; offset < n_samples; offset += batch_size) {
    IndexT current_batch_size = std::min(batch_size, n_samples - offset);

    raft::copy(batch_data.data_handle(),
               X.data_handle() + offset * n_features,
               current_batch_size * n_features,
               stream);

    auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
      batch_data.data_handle(), current_batch_size, n_features);
    auto minClusterAndDistance_view =
      raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
        minClusterAndDistance.data_handle(), current_batch_size);

    if (metric == cuvs::distance::DistanceType::L2Expanded ||
        metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
        L2NormBatch.data_handle(), batch_data.data_handle(), n_features, current_batch_size, stream);
    }

    auto centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      centroids.data_handle(), n_clusters, n_features);
    auto L2NormBatch_const = raft::make_device_vector_view<const DataT, IndexT>(
      L2NormBatch.data_handle(), current_batch_size);

    cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<DataT, IndexT>(
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

    cuvs::cluster::kmeans::detail::computeClusterCost(
      handle,
      minClusterAndDistance_view,
      workspace,
      raft::make_device_scalar_view(clusterCostD.data()),
      raft::value_op{},
      raft::add_op{});

    inertia[0] += clusterCostD.value(stream);
  }

  RAFT_LOG_DEBUG("KMeans batched: Completed with inertia=%f", inertia[0]);
}

}  // namespace cuvs::cluster::kmeans::batched::detail

