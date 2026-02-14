/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "kmeans.cuh"
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
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
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
#include <cstring>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Sample data from host to device for initialization, with optional type conversion
 *
 * @tparam T      Input data type
 * @tparam MathT  Computation/output type
 * @tparam IdxT   Index type
 * @tparam MappingOpT Mapping operator (T -> MathT)
 */
template <typename T, typename MathT, typename IdxT, typename MappingOpT>
void prepare_init_sample(raft::resources const& handle,
                         raft::host_matrix_view<const T, IdxT> X,
                         raft::device_matrix_view<MathT, IdxT> X_sample,
                         uint64_t seed,
                         MappingOpT mapping_op)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_samples_out  = X_sample.extent(0);

  std::mt19937 gen(seed);
  std::vector<IdxT> indices(n_samples);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), gen);

  // Sample raw T data to host buffer
  std::vector<T> host_sample(n_samples_out * n_features);

#pragma omp parallel for
  for (IdxT i = 0; i < static_cast<IdxT>(n_samples_out); i++) {
    IdxT src_idx = indices[i];
    std::memcpy(host_sample.data() + i * n_features,
                X.data_handle() + src_idx * n_features,
                n_features * sizeof(T));
  }

  if constexpr (std::is_same_v<T, MathT>) {
    // Same type: direct copy
    raft::copy(X_sample.data_handle(), host_sample.data(), host_sample.size(), stream);
  } else {
    // Different types: copy raw, then convert on GPU
    auto raw_sample = raft::make_device_matrix<T, IdxT>(handle, n_samples_out, n_features);
    raft::copy(raw_sample.data_handle(), host_sample.data(), host_sample.size(), stream);
    raft::linalg::unaryOp(
      X_sample.data_handle(), raw_sample.data_handle(), host_sample.size(), mapping_op, stream);
  }
}

/**
 * @brief Initialize centroids using k-means++ on a sample of the host data
 *
 * @tparam T      Input data type
 * @tparam MathT  Computation/centroid type
 * @tparam IdxT   Index type
 * @tparam MappingOpT Mapping operator (T -> MathT)
 */
template <typename T, typename MathT, typename IdxT, typename MappingOpT>
void init_centroids_from_host_sample(raft::resources const& handle,
                                     const cuvs::cluster::kmeans::params& params,
                                     raft::host_matrix_view<const T, IdxT> X,
                                     raft::device_matrix_view<MathT, IdxT> centroids,
                                     rmm::device_uvector<char>& workspace,
                                     MappingOpT mapping_op)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;

  // Sample size for initialization: at least 3 * n_clusters, but not more than n_samples
  size_t init_sample_size =
    std::min(static_cast<size_t>(n_samples),
             std::max(static_cast<size_t>(3 * n_clusters), static_cast<size_t>(10000)));

  RAFT_LOG_DEBUG("KMeans batched: sampling %zu points for initialization", init_sample_size);

  // Sample data from host to device (with conversion if needed)
  auto init_sample = raft::make_device_matrix<MathT, IdxT>(handle, init_sample_size, n_features);
  prepare_init_sample(handle, X, init_sample.view(), params.rng_state.seed, mapping_op);

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    auto init_sample_view = raft::make_device_matrix_view<const MathT, IdxT>(
      init_sample.data_handle(), init_sample_size, n_features);

    if (params.oversampling_factor == 0) {
      cuvs::cluster::kmeans::detail::kmeansPlusPlus<MathT, IdxT>(
        handle, params, init_sample_view, centroids, workspace);
    } else {
      cuvs::cluster::kmeans::detail::initScalableKMeansPlusPlus<MathT, IdxT>(
        handle, params, init_sample_view, centroids, workspace);
    }
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    // Just use the first n_clusters samples
    raft::copy(centroids.data_handle(), init_sample.data_handle(), n_clusters * n_features, stream);
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

  // Get workspace from handle
  auto workspace = rmm::device_uvector<char>(
    batch_data.extent(0), stream, raft::resource::get_workspace_resource(handle));

  // Temporary buffers for this batch's partial results
  auto batch_sums   = raft::make_device_matrix<MathT, IdxT>(handle, n_clusters, n_features);
  auto batch_counts = raft::make_device_vector<MathT, IdxT>(handle, n_clusters);

  // Extract cluster labels from KeyValuePair
  cuvs::cluster::kmeans::detail::KeyValueIndexOp<IdxT, MathT> conversion_op;
  thrust::transform_iterator<cuvs::cluster::kmeans::detail::KeyValueIndexOp<IdxT, MathT>,
                             const raft::KeyValuePair<IdxT, MathT>*>
    labels_itr(minClusterAndDistance.data_handle(), conversion_op);

  // Compute weighted sums and counts per cluster for this batch
  cuvs::cluster::kmeans::detail::compute_centroid_adjustments(handle,
                                                              batch_data,
                                                              sample_weights,
                                                              labels_itr,
                                                              static_cast<IdxT>(n_clusters),
                                                              batch_sums.view(),
                                                              batch_counts.view(),
                                                              workspace);

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
 * @brief Update centroids using mini-batch online learning
 *
 * Uses the online update formula:
 *   learning_rate[k] = batch_count[k] / (total_count[k] + batch_count[k])
 *   centroid[k] = centroid[k] + learning_rate[k] * (batch_mean[k] - centroid[k])
 *
 * This is equivalent to a weighted average where total_count tracks cumulative weight.
 */
template <typename MathT, typename IdxT>
void minibatch_update_centroids(raft::resources const& handle,
                                raft::device_matrix_view<MathT, IdxT> centroids,
                                raft::device_matrix_view<const MathT, IdxT> batch_sums,
                                raft::device_vector_view<const MathT, IdxT> batch_counts,
                                raft::device_vector_view<MathT, IdxT> total_counts)
{
  auto n_clusters = centroids.extent(0);
  auto n_features = centroids.extent(1);

  // Compute batch means: batch_mean = batch_sums / batch_counts
  auto batch_means = raft::make_device_matrix<MathT, IdxT>(handle, n_clusters, n_features);
  raft::copy(batch_means.data_handle(),
             batch_sums.data_handle(),
             batch_sums.size(),
             raft::resource::get_cuda_stream(handle));

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
    handle,
    raft::make_const_mdspan(batch_means.view()),
    batch_counts,
    batch_means.view(),
    raft::div_checkzero_op{});

  // Step 1: Update total_counts = total_counts + batch_counts
  raft::linalg::add(handle, raft::make_const_mdspan(total_counts), batch_counts, total_counts);

  // Step 2: Compute learning rates: lr = batch_count / total_count (after update)
  auto learning_rates = raft::make_device_vector<MathT, IdxT>(handle, n_clusters);
  raft::linalg::map(handle,
                    learning_rates.view(),
                    raft::div_checkzero_op{},
                    batch_counts,
                    raft::make_const_mdspan(total_counts));

  // Update centroids: centroid = centroid + lr * (batch_mean - centroid)
  //                            = (1 - lr) * centroid + lr * batch_mean
  // Using matrix_vector_op to scale each row by (1 - lr), then add lr * batch_mean
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
    handle,
    raft::make_const_mdspan(centroids),
    raft::make_const_mdspan(learning_rates.view()),
    centroids,
    [] __device__(MathT centroid_val, MathT lr) { return (MathT{1} - lr) * centroid_val; });

  // Add lr * batch_mean to centroids
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
    handle,
    raft::make_const_mdspan(batch_means.view()),
    raft::make_const_mdspan(learning_rates.view()),
    batch_means.view(),
    [] __device__(MathT mean_val, MathT lr) { return lr * mean_val; });

  // centroids += lr * batch_means
  raft::linalg::add(handle,
                    raft::make_const_mdspan(centroids),
                    raft::make_const_mdspan(batch_means.view()),
                    centroids);
}

/**
 * @brief Finalize centroids by dividing accumulated sums by counts
 */
template <typename MathT, typename IdxT>
void finalize_centroids(raft::resources const& handle,
                        raft::device_matrix_view<const MathT, IdxT> centroid_sums,
                        raft::device_vector_view<const MathT, IdxT> cluster_counts,
                        raft::device_matrix_view<const MathT, IdxT> old_centroids,
                        raft::device_matrix_view<MathT, IdxT> new_centroids)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_clusters     = new_centroids.extent(0);
  auto n_features     = new_centroids.extent(1);

  // Copy sums to new_centroids first
  raft::copy(
    new_centroids.data_handle(), centroid_sums.data_handle(), centroid_sums.size(), stream);

  // Divide by counts: new_centroids[i] = centroid_sums[i] / cluster_counts[i]
  // When count is 0, set to 0 (will be fixed below)
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(handle,
                                                             raft::make_const_mdspan(new_centroids),
                                                             cluster_counts,
                                                             new_centroids,
                                                             raft::div_checkzero_op{});

  // Copy old centroids to new centroids where cluster_counts[i] == 0
  cub::ArgIndexInputIterator<const MathT*> itr_wt(cluster_counts.data_handle());
  raft::matrix::gather_if(
    old_centroids.data_handle(),
    static_cast<int>(old_centroids.extent(1)),
    static_cast<int>(old_centroids.extent(0)),
    itr_wt,
    itr_wt,
    static_cast<int>(cluster_counts.size()),
    new_centroids.data_handle(),
    [=] __device__(raft::KeyValuePair<ptrdiff_t, MathT> map) {
      return map.value == MathT{0};  // predicate: copy when count is 0
    },
    raft::key_op{},
    stream);
}

/**
 * @brief Main fit function for batched k-means with host data
 *
 * This is a unified function that handles both same-type (T == MathT) and
 * mixed-type (T != MathT) cases, following the kmeans_balanced pattern.
 *
 * @tparam T         Input data type (float, double, uint8_t, int8_t, half)
 * @tparam MathT     Computation/centroid type (typically float)
 * @tparam IdxT      Index type (int, int64_t)
 * @tparam MappingOpT Mapping operator (T -> MathT)
 *
 * @param[in]     handle        RAFT resources handle
 * @param[in]     params        K-means parameters
 * @param[in]     X             Input data on HOST [n_samples x n_features]
 * @param[in]     batch_size    Number of samples to process per batch
 * @param[in]     sample_weight Optional weights per sample (on host, MathT type)
 * @param[inout]  centroids     Initial/output cluster centers [n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances to nearest centroid
 * @param[out]    n_iter        Number of iterations run
 * @param[in]     mapping_op    Mapping operator for T -> MathT conversion
 */
template <typename T, typename MathT, typename IdxT, typename MappingOpT>
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const T, IdxT> X,
         IdxT batch_size,
         std::optional<raft::host_vector_view<const MathT, IdxT>> sample_weight,
         raft::device_matrix_view<MathT, IdxT> centroids,
         raft::host_scalar_view<MathT> inertia,
         raft::host_scalar_view<IdxT> n_iter,
         MappingOpT mapping_op)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  RAFT_EXPECTS(batch_size > 0, "batch_size must be positive");
  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IdxT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");

  raft::default_logger().set_level(params.verbosity);

  RAFT_LOG_DEBUG("KMeans batched fit: n_samples=%zu, n_features=%zu, n_clusters=%d, batch_size=%zu",
                 static_cast<size_t>(n_samples),
                 static_cast<size_t>(n_features),
                 n_clusters,
                 static_cast<size_t>(batch_size));

  rmm::device_uvector<char> workspace(0, stream);

  // Initialize centroids from a sample of host data
  if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
    init_centroids_from_host_sample(handle, params, X, centroids, workspace, mapping_op);
  }

  // Allocate device buffers
  // For mixed types, we need a raw buffer for T and a converted buffer for MathT
  // For same types, we only need one buffer
  rmm::device_uvector<T> batch_data_raw(0, stream);
  if constexpr (!std::is_same_v<T, MathT>) {
    batch_data_raw.resize(batch_size * n_features, stream);
  }

  auto batch_data    = raft::make_device_matrix<MathT, IdxT>(handle, batch_size, n_features);
  auto batch_weights = raft::make_device_vector<MathT, IdxT>(handle, batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IdxT, MathT>, IdxT>(handle, batch_size);
  auto L2NormBatch = raft::make_device_vector<MathT, IdxT>(handle, batch_size);
  rmm::device_uvector<MathT> L2NormBuf_OR_DistBuf(0, stream);

  // Accumulators for centroid computation
  auto centroid_sums  = raft::make_device_matrix<MathT, IdxT>(handle, n_clusters, n_features);
  auto cluster_counts = raft::make_device_vector<MathT, IdxT>(handle, n_clusters);
  auto new_centroids  = raft::make_device_matrix<MathT, IdxT>(handle, n_clusters, n_features);

  // For mini-batch mode: track total counts for learning rate calculation
  auto total_counts = raft::make_device_vector<MathT, IdxT>(handle, n_clusters);

  // Host buffer for batch data
  std::vector<T> host_batch_buffer(batch_size * n_features);
  std::vector<MathT> host_weight_buffer(batch_size);

  // Cluster cost for convergence check
  rmm::device_scalar<MathT> clusterCostD(stream);
  MathT priorClusteringCost = 0;

  // Check update mode
  bool use_minibatch =
    (params.update_mode == cuvs::cluster::kmeans::params::CentroidUpdateMode::MiniBatch);

  RAFT_LOG_DEBUG("KMeans batched: update_mode=%s", use_minibatch ? "MiniBatch" : "FullBatch");

  // For mini-batch mode with random sampling, create index shuffle
  std::vector<IdxT> sample_indices(n_samples);
  std::iota(sample_indices.begin(), sample_indices.end(), 0);
  std::mt19937 rng(params.rng_state.seed);

  // Main iteration loop
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    RAFT_LOG_DEBUG("KMeans batched: Iteration %d", n_iter[0]);

    // For full-batch mode: zero accumulators at start of each iteration
    // For mini-batch mode: zero total_counts at start of each iteration
    if (!use_minibatch) {
      raft::matrix::fill(handle, centroid_sums.view(), MathT{0});
      raft::matrix::fill(handle, cluster_counts.view(), MathT{0});
    } else {
      // Mini-batch mode: zero total counts for learning rate calculation
      raft::matrix::fill(handle, total_counts.view(), MathT{0});
      // Shuffle sample indices for random batch selection
      std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    }

    // Save old centroids for convergence check
    raft::copy(new_centroids.data_handle(), centroids.data_handle(), centroids.size(), stream);

    MathT total_cost = 0;

    // Process all data in batches
    for (IdxT batch_idx = 0; batch_idx < n_samples; batch_idx += batch_size) {
      IdxT current_batch_size = std::min(batch_size, n_samples - batch_idx);

      // Copy batch data from host to device
      if (use_minibatch) {
        // Mini-batch: use shuffled indices for random sampling
#pragma omp parallel for
        for (IdxT i = 0; i < current_batch_size; ++i) {
          IdxT sample_idx = sample_indices[batch_idx + i];
          std::memcpy(host_batch_buffer.data() + i * n_features,
                      X.data_handle() + sample_idx * n_features,
                      n_features * sizeof(T));
        }

        if constexpr (std::is_same_v<T, MathT>) {
          raft::copy(batch_data.data_handle(),
                     host_batch_buffer.data(),
                     current_batch_size * n_features,
                     stream);
        } else {
          raft::copy(batch_data_raw.data(),
                     host_batch_buffer.data(),
                     current_batch_size * n_features,
                     stream);
          raft::linalg::unaryOp(batch_data.data_handle(),
                                batch_data_raw.data(),
                                current_batch_size * n_features,
                                mapping_op,
                                stream);
        }
      } else {
        // Full-batch: sequential access
        if constexpr (std::is_same_v<T, MathT>) {
          raft::copy(batch_data.data_handle(),
                     X.data_handle() + batch_idx * n_features,
                     current_batch_size * n_features,
                     stream);
        } else {
          raft::copy(batch_data_raw.data(),
                     X.data_handle() + batch_idx * n_features,
                     current_batch_size * n_features,
                     stream);
          raft::linalg::unaryOp(batch_data.data_handle(),
                                batch_data_raw.data(),
                                current_batch_size * n_features,
                                mapping_op,
                                stream);
        }
      }

      // Copy or set weights for this batch
      if (sample_weight) {
        if (use_minibatch) {
          for (IdxT i = 0; i < current_batch_size; ++i) {
            host_weight_buffer[i] = sample_weight->data_handle()[sample_indices[batch_idx + i]];
          }
          raft::copy(
            batch_weights.data_handle(), host_weight_buffer.data(), current_batch_size, stream);
        } else {
          raft::copy(batch_weights.data_handle(),
                     sample_weight->data_handle() + batch_idx,
                     current_batch_size,
                     stream);
        }
      } else {
        auto batch_weights_fill_view = raft::make_device_vector_view<MathT, IdxT>(
          batch_weights.data_handle(), current_batch_size);
        raft::matrix::fill(handle, batch_weights_fill_view, MathT{1});
      }

      // Create views for current batch size
      auto batch_data_view = raft::make_device_matrix_view<const MathT, IdxT>(
        batch_data.data_handle(), current_batch_size, n_features);
      auto batch_weights_view = raft::make_device_vector_view<const MathT, IdxT>(
        batch_weights.data_handle(), current_batch_size);
      auto minClusterAndDistance_view =
        raft::make_device_vector_view<raft::KeyValuePair<IdxT, MathT>, IdxT>(
          minClusterAndDistance.data_handle(), current_batch_size);

      // Compute L2 norms for batch if needed
      if (metric == cuvs::distance::DistanceType::L2Expanded ||
          metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        raft::linalg::rowNorm<raft::linalg::L2Norm, true>(L2NormBatch.data_handle(),
                                                          batch_data.data_handle(),
                                                          n_features,
                                                          current_batch_size,
                                                          stream);
      }

      // Find nearest centroid for each sample in batch
      auto centroids_const = raft::make_device_matrix_view<const MathT, IdxT>(
        centroids.data_handle(), n_clusters, n_features);
      auto L2NormBatch_const = raft::make_device_vector_view<const MathT, IdxT>(
        L2NormBatch.data_handle(), current_batch_size);

      cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<MathT, IdxT>(
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
        raft::make_device_vector_view<const raft::KeyValuePair<IdxT, MathT>, IdxT>(
          minClusterAndDistance.data_handle(), current_batch_size);

      if (use_minibatch) {
        // Mini-batch mode: zero batch accumulators before each batch
        raft::matrix::fill(handle, centroid_sums.view(), MathT{0});
        raft::matrix::fill(handle, cluster_counts.view(), MathT{0});
      }

      accumulate_batch_centroids<MathT, IdxT>(handle,
                                              batch_data_view,
                                              minClusterAndDistance_const,
                                              batch_weights_view,
                                              centroid_sums.view(),
                                              cluster_counts.view());

      if (use_minibatch) {
        // Mini-batch mode: update centroids immediately after each batch
        auto centroid_sums_const = raft::make_device_matrix_view<const MathT, IdxT>(
          centroid_sums.data_handle(), n_clusters, n_features);
        auto cluster_counts_const = raft::make_device_vector_view<const MathT, IdxT>(
          cluster_counts.data_handle(), n_clusters);

        minibatch_update_centroids<MathT, IdxT>(
          handle, centroids, centroid_sums_const, cluster_counts_const, total_counts.view());
      }

      // Accumulate cluster cost if checking convergence
      if (params.inertia_check) {
        cuvs::cluster::kmeans::detail::computeClusterCost(
          handle,
          minClusterAndDistance_view,
          workspace,
          raft::make_device_scalar_view(clusterCostD.data()),
          raft::value_op{},
          raft::add_op{});
        MathT batch_cost = clusterCostD.value(stream);
        total_cost += batch_cost;
      }
    }  // end batch loop

    if (!use_minibatch) {
      // Full-batch mode: finalize centroids after processing all batches
      auto centroids_const = raft::make_device_matrix_view<const MathT, IdxT>(
        centroids.data_handle(), n_clusters, n_features);
      auto centroid_sums_const = raft::make_device_matrix_view<const MathT, IdxT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto cluster_counts_const =
        raft::make_device_vector_view<const MathT, IdxT>(cluster_counts.data_handle(), n_clusters);

      finalize_centroids<MathT, IdxT>(
        handle, centroid_sums_const, cluster_counts_const, centroids_const, centroids);
    }

    // Compute squared norm of change in centroids (compare to saved old centroids)
    auto sqrdNorm = raft::make_device_scalar<MathT>(handle, MathT{0});
    raft::linalg::mapThenSumReduce(sqrdNorm.data_handle(),
                                   centroids.size(),
                                   raft::sqdiff_op{},
                                   stream,
                                   new_centroids.data_handle(),  // old centroids
                                   centroids.data_handle());     // new centroids

    MathT sqrdNormError = 0;
    raft::copy(&sqrdNormError, sqrdNorm.data_handle(), 1, stream);

    // Check convergence
    bool done = false;
    if (params.inertia_check) {
      if (n_iter[0] > 1) {
        MathT delta = total_cost / priorClusteringCost;
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
  for (IdxT offset = 0; offset < n_samples; offset += batch_size) {
    IdxT current_batch_size = std::min(batch_size, n_samples - offset);

    if constexpr (std::is_same_v<T, MathT>) {
      raft::copy(batch_data.data_handle(),
                 X.data_handle() + offset * n_features,
                 current_batch_size * n_features,
                 stream);
    } else {
      raft::copy(batch_data_raw.data(),
                 X.data_handle() + offset * n_features,
                 current_batch_size * n_features,
                 stream);
      raft::linalg::unaryOp(batch_data.data_handle(),
                            batch_data_raw.data(),
                            current_batch_size * n_features,
                            mapping_op,
                            stream);
    }

    auto batch_data_view = raft::make_device_matrix_view<const MathT, IdxT>(
      batch_data.data_handle(), current_batch_size, n_features);
    auto minClusterAndDistance_view =
      raft::make_device_vector_view<raft::KeyValuePair<IdxT, MathT>, IdxT>(
        minClusterAndDistance.data_handle(), current_batch_size);

    if (metric == cuvs::distance::DistanceType::L2Expanded ||
        metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
      raft::linalg::rowNorm<raft::linalg::L2Norm, true>(L2NormBatch.data_handle(),
                                                        batch_data.data_handle(),
                                                        n_features,
                                                        current_batch_size,
                                                        stream);
    }

    auto centroids_const = raft::make_device_matrix_view<const MathT, IdxT>(
      centroids.data_handle(), n_clusters, n_features);
    auto L2NormBatch_const = raft::make_device_vector_view<const MathT, IdxT>(
      L2NormBatch.data_handle(), current_batch_size);

    cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<MathT, IdxT>(
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

  RAFT_LOG_DEBUG("KMeans batched: Completed with inertia=%f", static_cast<double>(inertia[0]));
}

}  // namespace cuvs::cluster::kmeans::detail
