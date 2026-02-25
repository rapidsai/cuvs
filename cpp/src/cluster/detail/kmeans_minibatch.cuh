/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../neighbors/detail/ann_utils.cuh"
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

  // Step 1: centroid[k] *= old_total_counts[k]  (undo mean → get running sum)
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(handle,
                                                             raft::make_const_mdspan(centroids),
                                                             raft::make_const_mdspan(total_counts),
                                                             centroids,
                                                             raft::mul_op{});

  // Step 2: centroid[k] += batch_sums[k]
  raft::linalg::add(
    handle, raft::make_const_mdspan(centroids), raft::make_const_mdspan(batch_sums), centroids);

  // Step 3: total_counts[k] += batch_counts[k]
  raft::linalg::add(handle, raft::make_const_mdspan(total_counts), batch_counts, total_counts);

  // Step 4: centroid[k] /= total_counts[k]  (back to mean)
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(handle,
                                                             raft::make_const_mdspan(centroids),
                                                             raft::make_const_mdspan(total_counts),
                                                             centroids,
                                                             raft::div_checkzero_op{});

  // --- Reassignment logic: reassign low-count clusters to random samples ---
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

      // Reset total_counts for reassigned clusters to min of non-reassigned clusters
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
 * @brief Fit mini-batch k-means from host data.
 *
 * Randomly samples mini-batches from the host dataset each step, assigns
 * samples to their nearest centroid, and updates centroids using an online
 * learning rule.  Converges based on smoothed inertia (EWA) and center shift.
 *
 * @note When sample weights are provided they are used as sampling
 *       probabilities (via std::discrete_distribution) so each mini-batch is
 *       weight-aware.  Unit weights are passed to the centroid update to
 *       avoid double-weighting (matching scikit-learn).
 */
template <typename T, typename IdxT>
void minibatch_fit(raft::resources const& handle,
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

  RAFT_LOG_DEBUG(
    "KMeans minibatch fit: n_samples=%zu, n_features=%zu, n_clusters=%d, batch_size=%zu",
    static_cast<size_t>(n_samples),
    static_cast<size_t>(n_features),
    n_clusters,
    static_cast<size_t>(batch_size));

  rmm::device_uvector<char> workspace(0, stream);

  // --- Initialization ---
  if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
    init_centroids_from_host_sample(handle, params, X, centroids, workspace);
  }

  // Pre-allocate batch-sized device buffers
  auto batch_data    = raft::make_device_matrix<T, IdxT>(handle, batch_size, n_features);
  auto batch_weights = raft::make_device_vector<T, IdxT>(handle, batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IdxT, T>, IdxT>(handle, batch_size);
  auto L2NormBatch = raft::make_device_vector<T, IdxT>(handle, batch_size);
  rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);

  auto centroid_sums  = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);
  auto cluster_counts = raft::make_device_vector<T, IdxT>(handle, n_clusters);

  rmm::device_scalar<T> clusterCostD(stream);

  // Mini-batch specific state
  auto total_counts      = raft::make_device_vector<T, IdxT>(handle, n_clusters);
  auto host_batch_buffer = raft::make_host_matrix<T, IdxT>(batch_size, n_features);
  auto batch_indices     = raft::make_host_vector<IdxT, IdxT>(batch_size);

  std::mt19937 rng(params.rng_state.seed);
  std::uniform_int_distribution<IdxT> uniform_dist(0, n_samples - 1);

  // Weighted sampling setup
  std::discrete_distribution<IdxT> weighted_dist;
  bool use_weighted_sampling = false;
  if (sample_weight) {
    std::vector<double> weights(sample_weight->data_handle(),
                                sample_weight->data_handle() + n_samples);
    weighted_dist         = std::discrete_distribution<IdxT>(weights.begin(), weights.end());
    use_weighted_sampling = true;
  }

  // Convergence tracking
  T ewa_inertia        = T{0};
  T ewa_inertia_min    = T{0};
  int no_improvement   = 0;
  bool ewa_initialized = false;
  auto prev_centroids  = raft::make_device_matrix<T, IdxT>(handle, n_clusters, n_features);

  raft::matrix::fill(handle, total_counts.view(), T{0});
  // Since sampling is weight-aware (via discrete_distribution), we use unit weights
  // in the centroid update to avoid double weighting (matching scikit-learn).
  raft::matrix::fill(handle, batch_weights.view(), T{1});

  IdxT n_steps = (params.max_iter * n_samples) / batch_size;

  for (n_iter[0] = 1; n_iter[0] <= n_steps; ++n_iter[0]) {
    RAFT_LOG_DEBUG("KMeans minibatch: Step %d/%d", n_iter[0], n_steps);

    IdxT current_batch_size = batch_size;

    // Sample mini-batch indices
    for (IdxT i = 0; i < current_batch_size; ++i) {
      batch_indices.data_handle()[i] =
        use_weighted_sampling ? weighted_dist(rng) : uniform_dist(rng);
    }

    // Gather batch data on host
#pragma omp parallel for
    for (IdxT i = 0; i < current_batch_size; ++i) {
      IdxT sample_idx = batch_indices.data_handle()[i];
      std::memcpy(host_batch_buffer.data_handle() + i * n_features,
                  X.data_handle() + sample_idx * n_features,
                  n_features * sizeof(T));
    }

    // Copy batch to device
    raft::copy(batch_data.data_handle(),
               host_batch_buffer.data_handle(),
               current_batch_size * n_features,
               stream);

    auto batch_data_view = raft::make_device_matrix_view<const T, IdxT>(
      batch_data.data_handle(), current_batch_size, n_features);
    auto batch_weights_view_const =
      raft::make_device_vector_view<const T, IdxT>(batch_weights.data_handle(), current_batch_size);
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

    auto centroids_const =
      raft::make_device_matrix_view<const T, IdxT>(centroids.data_handle(), n_clusters, n_features);
    auto L2NormBatch_const =
      raft::make_device_vector_view<const T, IdxT>(L2NormBatch.data_handle(), current_batch_size);

    cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<T, IdxT>(handle,
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
                                        params.minibatch.reassignment_ratio,
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
      if (params.tol > 0.0 && centers_squared_diff <= params.tol) {
        RAFT_LOG_DEBUG(
          "KMeans minibatch: Converged (small centers change) at step %d/%d", n_iter[0], n_steps);
        break;
      }

      // Early stopping: lack of improvement in smoothed inertia
      if (params.minibatch.max_no_improvement > 0) {
        if (ewa_inertia < ewa_inertia_min) {
          no_improvement  = 0;
          ewa_inertia_min = ewa_inertia;
        } else {
          no_improvement++;
        }

        if (no_improvement >= params.minibatch.max_no_improvement) {
          RAFT_LOG_DEBUG(
            "KMeans minibatch: Converged (lack of improvement) at step %d/%d", n_iter[0], n_steps);
          break;
        }
      }
    } else {
      RAFT_LOG_DEBUG("KMeans minibatch step %d/%d: mean batch inertia: %f",
                     n_iter[0],
                     n_steps,
                     static_cast<double>(batch_inertia));
    }
  }

  // Final inertia computation (full pass over data)
  if (params.final_inertia_check) {
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
    RAFT_LOG_DEBUG("KMeans minibatch: Completed with inertia=%f", static_cast<double>(inertia[0]));
  } else {
    inertia[0] = 0;
    RAFT_LOG_DEBUG("KMeans minibatch: Completed (inertia computation skipped)");
  }
}

}  // namespace cuvs::cluster::kmeans::detail
