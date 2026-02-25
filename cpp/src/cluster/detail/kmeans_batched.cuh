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
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

namespace cuvs::cluster::kmeans::detail {

/**
 * @brief Sample data from host to device for initialization, with optional type conversion
 *
 * @tparam T      Input data type
 * @tparam IdxT   Index type
 */
template <typename T, typename IdxT>
void prepare_init_sample(raft::resources const& handle,
                         raft::host_matrix_view<const T, IdxT> X,
                         raft::device_matrix_view<T, IdxT> X_sample,
                         uint64_t seed)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_samples_out  = X_sample.extent(0);

  std::mt19937 gen(seed);
  std::uniform_int_distribution<IdxT> dist(0, n_samples - 1);

  // Generate n_samples_out unique random indices using rejection sampling
  // Since n_samples_out << n_samples, collisions are rare and this is O(n_samples_out)
  std::unordered_set<IdxT> selected_indices;
  selected_indices.reserve(n_samples_out);

  while (static_cast<IdxT>(selected_indices.size()) < n_samples_out) {
    IdxT idx = dist(gen);
    selected_indices.insert(idx);
  }

  std::vector<IdxT> indices(selected_indices.begin(), selected_indices.end());

  std::vector<T> host_sample(n_samples_out * n_features);
#pragma omp parallel for
  for (IdxT i = 0; i < static_cast<IdxT>(n_samples_out); i++) {
    IdxT src_idx = indices[i];
    std::memcpy(host_sample.data() + i * n_features,
                X.data_handle() + src_idx * n_features,
                n_features * sizeof(T));
  }

  raft::copy(X_sample.data_handle(), host_sample.data(), host_sample.size(), stream);
}

/**
 * @brief Initialize centroids using k-means++ on a sample of the host data
 *
 * @tparam T      Input data type
 * @tparam IdxT   Index type
 */
template <typename T, typename IdxT>
void init_centroids_from_host_sample(raft::resources const& handle,
                                     const cuvs::cluster::kmeans::params& params,
                                     raft::host_matrix_view<const T, IdxT> X,
                                     raft::device_matrix_view<T, IdxT> centroids,
                                     rmm::device_uvector<char>& workspace)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;

  size_t init_sample_size =
    std::min(static_cast<size_t>(n_samples),
             std::max(static_cast<size_t>(3 * n_clusters), static_cast<size_t>(10000)));
  RAFT_LOG_DEBUG("KMeans batched: sampling %zu points for initialization", init_sample_size);

  auto init_sample = raft::make_device_matrix<T, IdxT>(handle, init_sample_size, n_features);
  prepare_init_sample(handle, X, init_sample.view(), params.rng_state.seed);

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
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
    raft::copy(centroids.data_handle(), init_sample.data_handle(), n_clusters * n_features, stream);
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
 * @brief Update centroids using mini-batch online learning
 *
 * Updates centroids using the following formula (matching scikit-learn's implementation):
 *
 *   centroid_new[k] = (centroid_old[k] * old_total_counts[k] + batch_sums[k]) / total_counts[k]
 *
 * This is equivalent to the learning rate formula:
 *   learning_rate[k] = batch_counts[k] / total_counts[k]
 *   centroid[k] = centroid[k] * (1 - learning_rate[k]) + batch_means[k] * learning_rate[k]
 *
 * Optionally reassigns low-count clusters to random samples from the current batch.
 */
template <typename MathT, typename IdxT>
void minibatch_update_centroids(raft::resources const& handle,
                                raft::device_matrix_view<MathT, IdxT> centroids,
                                raft::device_matrix_view<const MathT, IdxT> batch_sums,
                                raft::device_vector_view<const MathT, IdxT> batch_counts,
                                raft::device_vector_view<MathT, IdxT> total_counts,
                                raft::device_matrix_view<const MathT, IdxT> batch_data,
                                double reassignment_ratio,
                                IdxT current_batch_size,
                                std::mt19937& rng)
{
  auto n_clusters     = centroids.extent(0);
  auto n_features     = centroids.extent(1);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(handle,
                                                             raft::make_const_mdspan(centroids),
                                                             raft::make_const_mdspan(total_counts),
                                                             centroids,
                                                             raft::mul_op{});

  raft::linalg::add(
    handle, raft::make_const_mdspan(centroids), raft::make_const_mdspan(batch_sums), centroids);

  raft::linalg::add(handle, raft::make_const_mdspan(total_counts), batch_counts, total_counts);

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(handle,
                                                             raft::make_const_mdspan(centroids),
                                                             raft::make_const_mdspan(total_counts),
                                                             centroids,
                                                             raft::div_checkzero_op{});

  // Reassignment logic: reassign low-count clusters to random samples from current batch
  if (reassignment_ratio > 0.0) {
    auto max_count_scalar     = raft::make_device_scalar<MathT>(handle, MathT{0});
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(nullptr,
                           temp_storage_bytes,
                           total_counts.data_handle(),
                           max_count_scalar.data_handle(),
                           n_clusters,
                           stream);
    rmm::device_uvector<char> temp_storage(temp_storage_bytes, stream);
    cub::DeviceReduce::Max(temp_storage.data(),
                           temp_storage_bytes,
                           total_counts.data_handle(),
                           max_count_scalar.data_handle(),
                           n_clusters,
                           stream);
    auto max_count_host = raft::make_host_scalar<MathT>(0);
    raft::copy(max_count_host.data_handle(), max_count_scalar.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    MathT max_count = max_count_host.data_handle()[0];

    MathT threshold     = static_cast<MathT>(reassignment_ratio) * max_count;
    auto reassign_flags = raft::make_device_vector<uint8_t, IdxT>(handle, n_clusters);

    raft::linalg::unaryOp(
      total_counts.data_handle(),
      reassign_flags.data_handle(),
      n_clusters,
      [=] __device__(MathT count) {
        return (count < threshold || count == MathT{0}) ? uint8_t{1} : uint8_t{0};
      },
      stream);

    auto num_reassign_scalar = raft::make_device_scalar<IdxT>(handle, IdxT{0});
    raft::linalg::mapThenSumReduce(num_reassign_scalar.data_handle(),
                                   n_clusters,
                                   raft::identity_op{},
                                   stream,
                                   reassign_flags.data_handle());
    auto num_reassign_host = raft::make_host_scalar<IdxT>(0);
    raft::copy(num_reassign_host.data_handle(), num_reassign_scalar.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    IdxT num_to_reassign = num_reassign_host.data_handle()[0];

    // Limit to 50% of batch size
    IdxT max_reassign = static_cast<IdxT>(0.5 * current_batch_size);
    if (num_to_reassign > max_reassign) {
      // Need to select only the worst ones - do sorting on device
      // First, get all cluster indices that need reassignment
      auto all_reassign_indices = raft::make_device_vector<IdxT, IdxT>(handle, num_to_reassign);
      auto counting_iter        = thrust::counting_iterator<IdxT>(0);
      thrust::device_ptr<uint8_t> flags_ptr(reassign_flags.data_handle());

      thrust::copy_if(raft::resource::get_thrust_policy(handle),
                      counting_iter,
                      counting_iter + n_clusters,
                      flags_ptr,
                      thrust::device_pointer_cast(all_reassign_indices.data_handle()),
                      [] __device__(uint8_t flag) { return flag == 1; });

      auto reassign_counts = raft::make_device_vector<MathT, IdxT>(handle, num_to_reassign);
      auto total_counts_matrix_view =
        raft::make_device_matrix_view<const MathT, IdxT>(total_counts.data_handle(), n_clusters, 1);
      auto reassign_indices_view = raft::make_device_vector_view<const IdxT, IdxT>(
        all_reassign_indices.data_handle(), num_to_reassign);
      auto reassign_counts_matrix_view = raft::make_device_matrix_view<MathT, IdxT>(
        reassign_counts.data_handle(), num_to_reassign, 1);
      raft::matrix::gather(
        handle, total_counts_matrix_view, reassign_indices_view, reassign_counts_matrix_view);

      thrust::sort_by_key(raft::resource::get_thrust_policy(handle),
                          reassign_counts.data_handle(),
                          reassign_counts.data_handle() + num_to_reassign,
                          all_reassign_indices.data_handle());

      raft::matrix::fill(handle, reassign_flags.view(), uint8_t{0});

      // Set flags only for worst max_reassign clusters
      auto worst_indices = raft::make_device_vector<IdxT, IdxT>(handle, max_reassign);
      raft::copy(
        worst_indices.data_handle(), all_reassign_indices.data_handle(), max_reassign, stream);

      auto flags_scatter = raft::make_device_vector<uint8_t, IdxT>(handle, max_reassign);
      raft::matrix::fill(handle, flags_scatter.view(), uint8_t{1});
      thrust::scatter(raft::resource::get_thrust_policy(handle),
                      flags_scatter.data_handle(),
                      flags_scatter.data_handle() + max_reassign,
                      worst_indices.data_handle(),
                      reassign_flags.data_handle());

      num_to_reassign = max_reassign;
    }

    if (num_to_reassign > 0) {
      // Get list of cluster indices to reassign
      auto reassign_indices = raft::make_device_vector<IdxT, IdxT>(handle, num_to_reassign);
      auto counting_iter    = thrust::counting_iterator<IdxT>(0);
      thrust::device_ptr<uint8_t> flags_ptr(reassign_flags.data_handle());

      thrust::copy_if(raft::resource::get_thrust_policy(handle),
                      counting_iter,
                      counting_iter + n_clusters,
                      flags_ptr,
                      thrust::device_pointer_cast(reassign_indices.data_handle()),
                      [] __device__(uint8_t flag) { return flag == 1; });

      auto actual_count_scalar = raft::make_device_scalar<IdxT>(handle, IdxT{0});
      raft::linalg::mapThenSumReduce(actual_count_scalar.data_handle(),
                                     n_clusters,
                                     raft::identity_op{},
                                     stream,
                                     reassign_flags.data_handle());
      auto actual_count_host = raft::make_host_scalar<IdxT>(0);
      raft::copy(actual_count_host.data_handle(), actual_count_scalar.data_handle(), 1, stream);
      raft::resource::sync_stream(handle, stream);
      num_to_reassign = actual_count_host.data_handle()[0];

      auto reassign_indices_host = raft::make_host_vector<IdxT, IdxT>(num_to_reassign);
      raft::copy(reassign_indices_host.data_handle(),
                 reassign_indices.data_handle(),
                 num_to_reassign,
                 stream);
      raft::resource::sync_stream(handle, stream);

      // Pick random samples from current batch (without replacement) on host
      std::uniform_int_distribution<IdxT> batch_dist(0, current_batch_size - 1);
      std::unordered_set<IdxT> selected_indices;
      selected_indices.reserve(num_to_reassign);

      while (static_cast<IdxT>(selected_indices.size()) < num_to_reassign) {
        IdxT idx = batch_dist(rng);
        selected_indices.insert(idx);
      }

      std::vector<IdxT> new_center_indices(selected_indices.begin(), selected_indices.end());

      for (IdxT i = 0; i < num_to_reassign; ++i) {
        IdxT cluster_idx = reassign_indices_host.data_handle()[i];
        IdxT sample_idx  = new_center_indices[i];
        raft::copy(centroids.data_handle() + cluster_idx * n_features,
                   batch_data.data_handle() + sample_idx * n_features,
                   n_features,
                   stream);
      }

      // Reset total_counts for reassigned clusters to min of non-reassigned clusters. Note that
      // this will affect the learning rate directly.
      auto masked_counts      = raft::make_device_vector<MathT, IdxT>(handle, n_clusters);
      auto total_counts_ptr   = total_counts.data_handle();
      auto reassign_flags_ptr = reassign_flags.data_handle();
      raft::linalg::map_offset(handle, masked_counts.view(), [=] __device__(IdxT k) {
        if (reassign_flags_ptr[k] == 0 && total_counts_ptr[k] > MathT{0}) {
          return total_counts_ptr[k];
        }
        return max_count;
      });

      auto min_non_reassigned_scalar = raft::make_device_scalar<MathT>(handle, max_count);
      size_t min_temp_storage_bytes  = 0;
      cub::DeviceReduce::Min(nullptr,
                             min_temp_storage_bytes,
                             masked_counts.data_handle(),
                             min_non_reassigned_scalar.data_handle(),
                             n_clusters,
                             stream);
      rmm::device_uvector<char> min_temp_storage(min_temp_storage_bytes, stream);
      cub::DeviceReduce::Min(min_temp_storage.data(),
                             min_temp_storage_bytes,
                             masked_counts.data_handle(),
                             min_non_reassigned_scalar.data_handle(),
                             n_clusters,
                             stream);
      auto min_non_reassigned_host = raft::make_host_scalar<MathT>(0);
      raft::copy(
        min_non_reassigned_host.data_handle(), min_non_reassigned_scalar.data_handle(), 1, stream);
      raft::resource::sync_stream(handle, stream);
      MathT min_non_reassigned = min_non_reassigned_host.data_handle()[0];
      if (min_non_reassigned == max_count) {
        min_non_reassigned = MathT{1};  // Fallback if all clusters were reassigned
      }

      // Update total_counts on device for reassigned clusters
      // reassign_indices_host is already available from earlier
      for (IdxT i = 0; i < num_to_reassign; ++i) {
        IdxT cluster_idx = reassign_indices_host.data_handle()[i];
        raft::copy(total_counts.data_handle() + cluster_idx, &min_non_reassigned, 1, stream);
      }

      RAFT_LOG_DEBUG("KMeans minibatch: Reassigned %zu cluster centers",
                     static_cast<size_t>(num_to_reassign));
    }
  }
}

/**
 * @brief Main fit function for batched k-means with host data
 *
 * @tparam T         Input data type (float, double)
 * @tparam IdxT      Index type (int, int64_t)
 *
 * @param[in]     handle        RAFT resources handle
 * @param[in]     params        K-means parameters
 * @param[in]     X             Input data on HOST [n_samples x n_features]
 * @param[in]     batch_size    Number of samples to process per batch
 * @param[in]     sample_weight Optional weights per sample (on host)
 * @param[inout]  centroids     Initial/output cluster centers [n_clusters x n_features]
 * @param[out]    inertia       Sum of squared distances to nearest centroid
 * @param[out]    n_iter        Number of iterations run
 *
 * @note For mini-batch mode: When sample weights are provided, they are used as sampling
 *       probabilities (normalized) to select minibatch samples. Unit weights are then passed
 *       to the centroid update to avoid double weighting (matching scikit-learn's approach).
 */
template <typename T, typename IdxT>
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const T, IdxT> X,
         IdxT batch_size,
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

  if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
    init_centroids_from_host_sample(handle, params, X, centroids, workspace);
  }

  bool use_minibatch =
    (params.batched.update_mode == cuvs::cluster::kmeans::params::CentroidUpdateMode::MiniBatch);
  RAFT_LOG_DEBUG("KMeans batched: update_mode=%s", use_minibatch ? "MiniBatch" : "FullBatch");

  auto batch_data    = raft::make_device_matrix<T, IdxT>(handle, batch_size, n_features);
  auto batch_weights = raft::make_device_vector<T, IdxT>(handle, batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IdxT, T>, IdxT>(handle, batch_size);
  auto L2NormBatch = raft::make_device_vector<T, IdxT>(handle, batch_size);
  rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);

  auto centroid_sums  = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);
  auto cluster_counts = raft::make_device_vector<T, IdxT>(handle, n_clusters);
  auto new_centroids  = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);

  rmm::device_scalar<T> clusterCostD(stream);
  T priorClusteringCost = 0;

  // Mini-batch only state
  auto total_counts = raft::make_device_vector<T, IdxT>(handle, use_minibatch ? n_clusters : 0);
  auto host_batch_buffer = use_minibatch ? raft::make_host_matrix<T, IdxT>(batch_size, n_features)
                                         : raft::make_host_matrix<T, IdxT>(0, n_features);
  auto batch_indices     = use_minibatch ? raft::make_host_vector<IdxT, IdxT>(batch_size)
                                         : raft::make_host_vector<IdxT, IdxT>(0);
  std::mt19937 rng(params.rng_state.seed);
  std::uniform_int_distribution<IdxT> uniform_dist(0, n_samples - 1);
  // Weighted sampling: if sample weights are provided, use them as sampling probabilities.
  // Since the sampling is weight-aware, we pass unit weights to the centroid update
  // to avoid accounting for the weights twice (matching scikit-learn's approach).
  std::discrete_distribution<IdxT> weighted_dist;
  bool use_weighted_sampling = false;
  if (use_minibatch && sample_weight) {
    std::vector<double> weights(sample_weight->data_handle(),
                                sample_weight->data_handle() + n_samples);
    weighted_dist         = std::discrete_distribution<IdxT>(weights.begin(), weights.end());
    use_weighted_sampling = true;
  }
  IdxT n_steps = params.max_iter;

  // Mini-batch convergence tracking
  T ewa_inertia        = T{0};
  T ewa_inertia_min    = T{0};
  int no_improvement   = 0;
  bool ewa_initialized = false;
  auto prev_centroids  = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);

  if (use_minibatch) {
    raft::matrix::fill(handle, total_counts.view(), T{0});
    // Fill once before the loop since batch_size is constant.
    raft::matrix::fill(handle, batch_weights.view(), T{1});
    n_steps = (params.max_iter * n_samples) / batch_size;
  }

  for (n_iter[0] = 1; n_iter[0] <= n_steps; ++n_iter[0]) {
    RAFT_LOG_DEBUG("KMeans batched: Iteration %d", n_iter[0]);

    raft::copy(new_centroids.data_handle(), centroids.data_handle(), centroids.size(), stream);

    T total_cost = 0;

    if (use_minibatch) {
      IdxT current_batch_size = batch_size;

      for (IdxT i = 0; i < current_batch_size; ++i) {
        batch_indices.data_handle()[i] =
          use_weighted_sampling ? weighted_dist(rng) : uniform_dist(rng);
      }

#pragma omp parallel for
      for (IdxT i = 0; i < current_batch_size; ++i) {
        IdxT sample_idx = batch_indices.data_handle()[i];
        std::memcpy(host_batch_buffer.data_handle() + i * n_features,
                    X.data_handle() + sample_idx * n_features,
                    n_features * sizeof(T));
      }

      raft::copy(batch_data.data_handle(),
                 host_batch_buffer.data_handle(),
                 current_batch_size * n_features,
                 stream);

      auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
        batch_data.data_handle(), current_batch_size, n_features);
      auto batch_weights_view_const = raft::make_device_vector_view<const T, IdxT>(
        batch_weights.data_handle(), current_batch_size);
      auto minClusterAndDistance_view =
        raft::make_device_vector_view<raft::KeyValuePair<IdxT, T>, IdxT>(
          minClusterAndDistance.data_handle(), current_batch_size);

      if (metric == cuvs::distance::DistanceType::L2Expanded ||
          metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        raft::linalg::rowNorm<raft::linalg::L2Norm, true>(L2NormBatch.data_handle(),
                                                          batch_data.data_handle(),
                                                          n_features,
                                                          current_batch_size,
                                                          stream);
      }

      // Save centroids before update for convergence check
      raft::copy(prev_centroids.data_handle(), centroids.data_handle(), centroids.size(), stream);

      auto centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        centroids.data_handle(), n_clusters, n_features);
      auto L2NormBatch_const =
        raft::make_device_vector_view<const T, IdxT>(L2NormBatch.data_handle(), current_batch_size);

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

      // Compute batch inertia (normalized by batch_size for comparison)
      T batch_inertia = 0;
      cuvs::cluster::kmeans::detail::computeClusterCost(
        handle,
        minClusterAndDistance_view,
        workspace,
        raft::make_device_scalar_view(clusterCostD.data()),
        raft::value_op{},
        raft::add_op{});
      auto clusterCost_host = raft::make_host_scalar<T>(0);
      raft::copy(clusterCost_host.data_handle(), clusterCostD.data(), 1, stream);
      raft::resource::sync_stream(handle, stream);
      batch_inertia = clusterCost_host.data_handle()[0] / static_cast<T>(current_batch_size);

      raft::matrix::fill(handle, centroid_sums.view(), T{0});
      raft::matrix::fill(handle, cluster_counts.view(), T{0});

      auto minClusterAndDistance_const = raft::make_const_mdspan(minClusterAndDistance_view);

      accumulate_batch_centroids<T, IdxT>(handle,
                                          batch_data_view,
                                          minClusterAndDistance_const,
                                          batch_weights_view_const,
                                          centroid_sums.view(),
                                          cluster_counts.view());

      auto centroid_sums_const = raft::make_device_matrix_view<const T, IdxT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto cluster_counts_const =
        raft::make_device_vector_view<const T, IdxT>(cluster_counts.data_handle(), n_clusters);

      minibatch_update_centroids<T, IdxT>(handle,
                                          centroids,
                                          centroid_sums_const,
                                          cluster_counts_const,
                                          total_counts.view(),
                                          batch_data_view,
                                          params.batched.minibatch.reassignment_ratio,
                                          current_batch_size,
                                          rng);

      // Compute squared difference of centers (for convergence check)
      auto sqrdNorm = raft::make_device_scalar<T>(handle, T{0});
      raft::linalg::mapThenSumReduce(sqrdNorm.data_handle(),
                                     centroids.size(),
                                     raft::sqdiff_op{},
                                     stream,
                                     prev_centroids.data_handle(),
                                     centroids.data_handle());
      T centers_squared_diff = 0;
      raft::copy(&centers_squared_diff, sqrdNorm.data_handle(), 1, stream);
      raft::resource::sync_stream(handle, stream);

      // Skip first step (inertia from initialization)
      if (n_iter[0] > 1) {
        // Update Exponentially Weighted Average of inertia
        T alpha = static_cast<T>(current_batch_size * 2.0) / static_cast<T>(n_samples + 1);
        alpha   = std::min(alpha, T{1});

        if (!ewa_initialized) {
          ewa_inertia     = batch_inertia;
          ewa_inertia_min = batch_inertia;
          ewa_initialized = true;
        } else {
          ewa_inertia = ewa_inertia * (T{1} - alpha) + batch_inertia * alpha;
        }

        RAFT_LOG_DEBUG(
          "KMeans minibatch step %d/%d: batch_inertia=%f, ewa_inertia=%f, centers_squared_diff=%f",
          n_iter[0],
          n_steps,
          static_cast<double>(batch_inertia),
          static_cast<double>(ewa_inertia),
          static_cast<double>(centers_squared_diff));

        // Early stopping: absolute tolerance on squared change of centers
        // Disabled if tol == 0.0
        if (params.tol > 0.0 && centers_squared_diff <= params.tol) {
          RAFT_LOG_DEBUG(
            "KMeans minibatch: Converged (small centers change) at step %d/%d", n_iter[0], n_steps);
          break;
        }

        // Early stopping: lack of improvement in smoothed inertia
        // Disabled if max_no_improvement == 0
        if (params.batched.minibatch.max_no_improvement > 0) {
          if (ewa_inertia < ewa_inertia_min) {
            no_improvement  = 0;
            ewa_inertia_min = ewa_inertia;
          } else {
            no_improvement++;
          }

          if (no_improvement >= params.batched.minibatch.max_no_improvement) {
            RAFT_LOG_DEBUG("KMeans minibatch: Converged (lack of improvement) at step %d/%d",
                           n_iter[0],
                           n_steps);
            break;
          }
        }
      } else {
        RAFT_LOG_DEBUG("KMeans minibatch step %d/%d: mean batch inertia: %f",
                       n_iter[0],
                       n_steps,
                       static_cast<double>(batch_inertia));
      }
    } else {
      raft::matrix::fill(handle, centroid_sums.view(), T{0});
      raft::matrix::fill(handle, cluster_counts.view(), T{0});

      auto centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        centroids.data_handle(), n_clusters, n_features);

      using namespace cuvs::spatial::knn::detail::utils;
      batch_load_iterator<T> data_batches(
        X.data_handle(), n_samples, n_features, batch_size, stream);

      for (const auto& data_batch : data_batches) {
        IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
          data_batch.data(), current_batch_size, n_features);

        auto batch_weights_fill_view =
          raft::make_device_vector_view<T, IdxT>(batch_weights.data_handle(), current_batch_size);
        if (sample_weight) {
          raft::copy(batch_weights.data_handle(),
                     sample_weight->data_handle() + data_batch.offset(),
                     current_batch_size,
                     stream);
        } else {
          raft::matrix::fill(handle, batch_weights_fill_view, T{1});
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

        if (params.inertia_check) {
          cuvs::cluster::kmeans::detail::computeClusterCost(
            handle,
            minClusterAndDistance.view(),
            workspace,
            raft::make_device_scalar_view(clusterCostD.data()),
            raft::value_op{},
            raft::add_op{});
          auto clusterCost_host = raft::make_host_scalar<T>(0);
          raft::copy(clusterCost_host.data_handle(), clusterCostD.data(), 1, stream);
          raft::resource::sync_stream(handle, stream);
          total_cost += clusterCost_host.data_handle()[0];
        }
      }

      auto centroid_sums_const = raft::make_device_matrix_view<const T, IdxT>(
        centroid_sums.data_handle(), n_clusters, n_features);
      auto cluster_counts_const =
        raft::make_device_vector_view<const T, IdxT>(cluster_counts.data_handle(), n_clusters);

      finalize_centroids<T, IdxT>(
        handle, centroid_sums_const, cluster_counts_const, centroids_const, new_centroids.view());
    }

    // Convergence check for full-batch mode only
    if (!use_minibatch) {
      auto sqrdNorm = raft::make_device_scalar<T>(handle, T{0});
      raft::linalg::mapThenSumReduce(sqrdNorm.data_handle(),
                                     centroids.size(),
                                     raft::sqdiff_op{},
                                     stream,
                                     new_centroids.data_handle(),
                                     centroids.data_handle());

      raft::copy(centroids.data_handle(), new_centroids.data_handle(), centroids.size(), stream);

      T sqrdNormError = 0;
      raft::copy(&sqrdNormError, sqrdNorm.data_handle(), 1, stream);

      bool done = false;
      if (params.inertia_check && n_iter[0] > 1) {
        T delta = total_cost / priorClusteringCost;
        if (delta > 1 - params.tol) done = true;
        priorClusteringCost = total_cost;
      }

      raft::resource::sync_stream(handle, stream);
      if (sqrdNormError < params.tol) done = true;

      if (done) {
        RAFT_LOG_DEBUG("KMeans batched: Converged after %d iterations", n_iter[0]);
        break;
      }
    }
  }

  if (params.batched.final_inertia_check) {
    inertia[0] = 0;
    using namespace cuvs::spatial::knn::detail::utils;
    batch_load_iterator<T> data_batches(X.data_handle(), n_samples, n_features, batch_size, stream);

    for (const auto& data_batch : data_batches) {
      IdxT current_batch_size = static_cast<IdxT>(data_batch.size());

      auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
        data_batch.data(), current_batch_size, n_features);
      auto minClusterAndDistance_view =
        raft::make_device_vector_view<raft::KeyValuePair<IdxT, T>, IdxT>(
          minClusterAndDistance.data_handle(), current_batch_size);

      if (metric == cuvs::distance::DistanceType::L2Expanded ||
          metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
          L2NormBatch.data_handle(), data_batch.data(), n_features, current_batch_size, stream);
      }

      auto centroids_const = raft::make_device_matrix_view<const T, IdxT>(
        centroids.data_handle(), n_clusters, n_features);
      auto L2NormBatch_const =
        raft::make_device_vector_view<const T, IdxT>(L2NormBatch.data_handle(), current_batch_size);

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

      cuvs::cluster::kmeans::detail::computeClusterCost(
        handle,
        minClusterAndDistance_view,
        workspace,
        raft::make_device_scalar_view(clusterCostD.data()),
        raft::value_op{},
        raft::add_op{});

      auto clusterCost_host = raft::make_host_scalar<T>(0);
      raft::copy(clusterCost_host.data_handle(), clusterCostD.data(), 1, stream);
      raft::resource::sync_stream(handle, stream);
      inertia[0] += clusterCost_host.data_handle()[0];
    }
    RAFT_LOG_DEBUG("KMeans batched: Completed with inertia=%f", static_cast<double>(inertia[0]));
  } else {
    inertia[0] = 0;
    RAFT_LOG_DEBUG("KMeans batched: Completed (inertia computation skipped)");
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
             IdxT batch_size,
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

  RAFT_EXPECTS(batch_size > 0, "batch_size must be positive");
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

    if (sample_weight) {
      raft::copy(batch_weights.data_handle(),
                 sample_weight->data_handle() + batch_idx,
                 current_batch_size,
                 stream);
    }

    auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
      batch_data.data_handle(), current_batch_size, n_features);

    T batch_inertia = 0;
    cuvs::cluster::kmeans::detail::kmeans_predict<T, IdxT>(
      handle,
      params,
      batch_data_view,
      batch_weights.view(),
      centroids,
      batch_labels.view(),
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
                 IdxT batch_size,
                 std::optional<raft::host_vector_view<const T, IdxT>> sample_weight,
                 raft::device_matrix_view<T, IdxT> centroids,
                 raft::host_vector_view<IdxT, IdxT> labels,
                 raft::host_scalar_view<T> inertia,
                 raft::host_scalar_view<IdxT> n_iter)
{
  T fit_inertia = 0;
  fit<T, IdxT>(handle,
               params,
               X,
               batch_size,
               sample_weight,
               centroids,
               raft::make_host_scalar_view(&fit_inertia),
               n_iter);

  auto centroids_const = raft::make_device_matrix_view<const T, IdxT>(
    centroids.data_handle(), centroids.extent(0), centroids.extent(1));

  predict<T, IdxT>(
    handle, params, X, batch_size, sample_weight, centroids_const, labels, false, inertia);
}

}  // namespace cuvs::cluster::kmeans::detail
