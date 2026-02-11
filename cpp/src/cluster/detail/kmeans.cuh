/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../core/nvtx.hpp"
#include "kmeans_common.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <optional>
#include <random>

namespace cuvs::cluster::kmeans::detail {

static const std::string kCuvsName = "cuvs";

// =========================================================
// Init functions
// =========================================================

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void init_random(raft::resources const& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::device_matrix_view<const DataT, IndexT> X,
                 raft::device_matrix_view<DataT, IndexT> centroids)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("init_random");
  auto n_clusters = params.n_clusters;
  cuvs::cluster::kmeans::detail::shuffle_and_gather<DataT, IndexT>(
    handle, X, centroids, n_clusters, params.rng_state.seed);
}

/*
 * @brief Selects 'n_clusters' samples from the input X using kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "k-means++: the advantages of careful seeding". 2007, Arthur, D. and Vassilvitskii, S.
 *        ACM-SIAM symposium on Discrete algorithms.
 *
 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: while |C| < k
 * 3:   Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
 * 4:   C = C U {x}
 * 5: end for
 */
template <typename DataT, typename IndexT>
void kmeans_plus_plus(raft::resources const& handle,
                      const cuvs::cluster::kmeans::params& params,
                      raft::device_matrix_view<const DataT, IndexT> X,
                      raft::device_matrix_view<DataT, IndexT> centroids_raw_data,
                      rmm::device_uvector<char>& workspace)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeans_plus_plus");
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  // number of seeding trials for each center (except the first)
  auto n_trials = 2 + static_cast<int>(std::ceil(log(n_clusters)));

  RAFT_LOG_DEBUG(
    "Run sequential k-means++ to select %d centroids from %d input samples "
    "(%d seeding trials per iterations)",
    n_clusters,
    n_samples,
    n_trials);

  auto data_batch_size = get_data_batch_size(params.batch_samples, n_samples);

  // temporary buffers
  auto indices              = raft::make_device_vector<IndexT, IndexT>(handle, n_trials);
  auto centroid_candidates  = raft::make_device_matrix<DataT, IndexT>(handle, n_trials, n_features);
  auto cost_per_candidate   = raft::make_device_vector<DataT, IndexT>(handle, n_trials);
  auto min_cluster_distance = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto dist_buffer          = raft::make_device_matrix<DataT, IndexT>(handle, n_trials, n_samples);

  rmm::device_uvector<DataT> l2_norm_buf_or_dist_buf(0, stream);
  rmm::device_scalar<DataT> cluster_cost(stream);
  rmm::device_scalar<cub::KeyValuePair<int, DataT>> min_cluster_index_and_distance(stream);

  // Device and matrix views
  raft::device_vector_view<IndexT, IndexT> indices_view(indices.data_handle(), n_trials);
  auto const_weights_view = raft::make_device_vector_view<const DataT, IndexT>(
    min_cluster_distance.data_handle(), n_samples);
  auto const_indices_view =
    raft::make_device_vector_view<const IndexT, IndexT>(indices.data_handle(), n_trials);
  auto const_x_view =
    raft::make_device_matrix_view<const DataT, IndexT>(X.data_handle(), n_samples, n_features);
  raft::device_matrix_view<DataT, IndexT> candidates_view(
    centroid_candidates.data_handle(), n_trials, n_features);

  // L2 norm of X: ||c||^2
  auto l2_norm_x = raft::make_device_vector<DataT, IndexT>(handle, n_samples);

  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      l2_norm_x.data_handle(), X.data_handle(), X.extent(1), X.extent(0), stream);
  }

  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  std::mt19937 gen(params.rng_state.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  // <<< Step-1 >>>: C <-- sample a point uniformly at random from X
  auto initial_centroid = raft::make_device_matrix_view<const DataT, IndexT>(
    X.data_handle() + dis(gen) * n_features, 1, n_features);
  int n_clusters_picked = 1;

  // store the chosen centroid in the buffer
  raft::copy(centroids_raw_data.data_handle(),
             initial_centroid.data_handle(),
             initial_centroid.size(),
             stream);

  //  C = initial set of centroids
  auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroids_raw_data.data_handle(), initial_centroid.extent(0), initial_centroid.extent(1));
  // <<< End of Step-1 >>>

  // Calculate cluster distance, d^2(x, C), for all the points x in X to the nearest centroid
  cuvs::cluster::kmeans::detail::min_cluster_distance_compute<DataT, IndexT>(
    handle,
    X,
    centroids,
    min_cluster_distance.view(),
    l2_norm_x.view(),
    l2_norm_buf_or_dist_buf,
    params.metric,
    params.batch_samples,
    params.batch_centroids,
    workspace);

  RAFT_LOG_DEBUG(" k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);

  // <<<< Step-2 >>> : while |C| < k
  while (n_clusters_picked < n_clusters) {
    // <<< Step-3 >>> : Sample x in X with probability p_x = d^2(x, C) / phi_X (C)
    // Choose 'n_trials' centroid candidates from X with probability proportional to the squared
    // distance to the nearest existing cluster

    raft::random::discrete(handle, rng, indices_view, const_weights_view);
    raft::matrix::gather(handle, const_x_view, const_indices_view, candidates_view);

    // Calculate pairwise distance between X and the centroid candidates
    // Output - pwd [n_trials x n_samples]
    auto pwd = dist_buffer.view();
    cuvs::cluster::kmeans::detail::pairwise_distance_kmeans<DataT, IndexT>(
      handle, centroid_candidates.view(), X, pwd, metric);

    // Update nearest cluster distance for each centroid candidate
    // Note pwd and min_dist_buf points to same buffer which currently holds pairwise distance
    // values. Outputs minDistanceBuf[n_trials x n_samples] where minDistance[i, :] contains updated
    // min_cluster_distance that includes candidate-i
    auto min_dist_buf = dist_buffer.view();
    raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_const_mdspan(pwd),
      raft::make_const_mdspan(min_cluster_distance.view()),
      min_dist_buf,
      raft::min_op{});

    // Calculate cost_per_candidate[n_trials] where cost_per_candidate[i] is the cluster cost when
    // using centroid candidate-i
    raft::linalg::reduce<true, true>(cost_per_candidate.data_handle(),
                                     min_dist_buf.data_handle(),
                                     min_dist_buf.extent(1),
                                     min_dist_buf.extent(0),
                                     static_cast<DataT>(0),
                                     stream);

    // Greedy Choice - Choose the candidate that has minimum cluster cost
    // ArgMin operation below identifies the index of minimum cost in cost_per_candidate
    {
      // Determine temporary device storage requirements
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::ArgMin(nullptr,
                                temp_storage_bytes,
                                cost_per_candidate.data_handle(),
                                min_cluster_index_and_distance.data(),
                                cost_per_candidate.extent(0),
                                stream);

      // Allocate temporary storage
      workspace.resize(temp_storage_bytes, stream);

      // Run argmin-reduction
      cub::DeviceReduce::ArgMin(workspace.data(),
                                temp_storage_bytes,
                                cost_per_candidate.data_handle(),
                                min_cluster_index_and_distance.data(),
                                cost_per_candidate.extent(0),
                                stream);

      int best_candidate_idx = -1;
      raft::copy(&best_candidate_idx, &min_cluster_index_and_distance.data()->key, 1, stream);
      raft::resource::sync_stream(handle);
      /// <<< End of Step-3 >>>

      /// <<< Step-4 >>>: C = C U {x}
      // Update minimum cluster distance corresponding to the chosen centroid candidate
      raft::copy(min_cluster_distance.data_handle(),
                 min_dist_buf.data_handle() + best_candidate_idx * n_samples,
                 n_samples,
                 stream);

      raft::copy(centroids_raw_data.data_handle() + n_clusters_picked * n_features,
                 centroid_candidates.data_handle() + best_candidate_idx * n_features,
                 n_features,
                 stream);

      ++n_clusters_picked;
      /// <<< End of Step-4 >>>
    }

    RAFT_LOG_DEBUG(" k-means++ - Sampled %d/%d centroids", n_clusters_picked, n_clusters);
  }  /// <<<< Step-5 >>>
}

/**
 *
 * @tparam DataT
 * @tparam IndexT
 * @param handle
 * @param[in] X input matrix (size n_samples, n_features)
 * @param[in] weight number of samples currently assigned to each centroid
 * @param[in] cur_centroids matrix of current centroids (size n_clusters, n_features)
 * @param[in] l2norm_x
 * @param[out] min_cluster_and_dist
 * @param[out] new_centroids
 * @param[out] new_weight
 * @param[inout] workspace
 */
template <typename DataT, typename IndexT, typename LabelsIterator>
void update_centroids(raft::resources const& handle,
                      raft::device_matrix_view<const DataT, IndexT, raft::row_major> X,
                      raft::device_vector_view<const DataT, IndexT> sample_weights,
                      raft::device_matrix_view<const DataT, IndexT, raft::row_major> centroids,

                      // TODO(snanditale): Figure out how to best wrap iterator types in mdspan
                      LabelsIterator cluster_labels,
                      raft::device_vector_view<DataT, IndexT> weight_per_cluster,
                      raft::device_matrix_view<DataT, IndexT, raft::row_major> new_centroids,
                      rmm::device_uvector<char>& workspace)
{
  auto n_clusters = centroids.extent(0);
  auto n_samples  = X.extent(0);

  workspace.resize(n_samples, raft::resource::get_cuda_stream(handle));

  // Calculates weighted sum of all the samples assigned to cluster-i and stores the
  // result in new_centroids[i]
  raft::linalg::reduce_rows_by_key(const_cast<DataT*>(X.data_handle()),
                                   X.extent(1),
                                   cluster_labels,
                                   sample_weights.data_handle(),
                                   workspace.data(),
                                   X.extent(0),
                                   X.extent(1),
                                   n_clusters,
                                   new_centroids.data_handle(),
                                   raft::resource::get_cuda_stream(handle));

  // Reduce weights by key to compute weight in each cluster
  raft::linalg::reduce_cols_by_key(sample_weights.data_handle(),
                                   cluster_labels,
                                   weight_per_cluster.data_handle(),
                                   static_cast<IndexT>(1),
                                   static_cast<IndexT>(sample_weights.extent(0)),
                                   static_cast<IndexT>(n_clusters),
                                   raft::resource::get_cuda_stream(handle));

  // Computes new_centroids[i] = new_centroids[i]/weight_per_cluster[i] where
  //   new_centroids[n_clusters x n_features] - 2D array, new_centroids[i] has sum of all the
  //   samples assigned to cluster-i
  //   weight_per_cluster[n_clusters] - 1D array, weight_per_cluster[i] contains sum of weights in
  //   cluster-i.
  // Note - when weight_per_cluster[i] is 0, new_centroids[i] is reset to 0
  raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
    handle,
    raft::make_const_mdspan(new_centroids),
    raft::make_const_mdspan(weight_per_cluster),
    new_centroids,
    raft::div_checkzero_op{});

  // copy centroids[i] to new_centroids[i] when weight_per_cluster[i] is 0
  cub::ArgIndexInputIterator<DataT*> itr_wt(weight_per_cluster.data_handle());
  raft::matrix::gather_if(
    const_cast<DataT*>(centroids.data_handle()),
    static_cast<int>(centroids.extent(1)),
    static_cast<int>(centroids.extent(0)),
    itr_wt,
    itr_wt,
    static_cast<int>(weight_per_cluster.size()),
    new_centroids.data_handle(),
    [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) -> bool {  // predicate
      // copy when the sum of weights in the cluster is 0
      return map.value == 0;
    },
    raft::key_op{},
    raft::resource::get_cuda_stream(handle));
}

// TODO(snanditale): Resizing is needed to use mdarray instead of rmm::device_uvector
template <typename DataT, typename IndexT>
void kmeans_fit_main(raft::resources const& handle,
                     const cuvs::cluster::kmeans::params& params,
                     raft::device_matrix_view<const DataT, IndexT> X,
                     raft::device_vector_view<const DataT, IndexT> weight,
                     raft::device_matrix_view<DataT, IndexT> centroids_raw_data,
                     raft::host_scalar_view<DataT> inertia,
                     raft::host_scalar_view<IndexT> n_iter,
                     rmm::device_uvector<char>& workspace)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeans_fit_main");
  raft::default_logger().set_level(params.verbosity);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto min_cluster_and_distance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> l2_norm_buf_or_dist_buf(0, stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  auto new_centroids = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  // temporary buffer to store weights per cluster, destructor releases the
  // resource
  auto wt_in_cluster = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);

  rmm::device_scalar<DataT> cluster_cost_d(stream);

  // L2 norm of X: ||x||^2
  auto l2_norm_x = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto l2normx_view =
    raft::make_device_vector_view<const DataT, IndexT>(l2_norm_x.data_handle(), n_samples);

  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      l2_norm_x.data_handle(), X.data_handle(), X.extent(1), X.extent(0), stream);
  }

  RAFT_LOG_DEBUG(
    "Calling KMeans.fit with %d samples of input data and the initialized "
    "cluster centers",
    n_samples);

  DataT prior_clustering_cost = 0;
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    RAFT_LOG_DEBUG(
      "KMeans.fit: Iteration-%d: fitting the model using the initialized "
      "cluster centers",
      n_iter[0]);

    auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
      centroids_raw_data.data_handle(), n_clusters, n_features);

    // computes min_cluster_and_distance[0:n_samples) where
    // min_cluster_and_distance[i] is a <key, value> pair where
    //   'key' is index to a sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    cuvs::cluster::kmeans::detail::min_cluster_and_distance_compute<DataT, IndexT>(
      handle,
      X,
      centroids,
      min_cluster_and_distance.view(),
      l2normx_view,
      l2_norm_buf_or_dist_buf,
      params.metric,
      params.batch_samples,
      params.batch_centroids,
      workspace);

    // Using TransformInputIteratorT to dereference an array of
    // raft::KeyValuePair and converting them to just return the Key to be used
    // in reduce_rows_by_key prims
    cuvs::cluster::kmeans::detail::key_value_index_op<IndexT, DataT> conversion_op;
    thrust::transform_iterator<cuvs::cluster::kmeans::detail::key_value_index_op<IndexT, DataT>,
                               raft::KeyValuePair<IndexT, DataT>*>
      itr(min_cluster_and_distance.data_handle(), conversion_op);

    update_centroids(handle,
                     X,
                     weight,
                     raft::make_device_matrix_view<const DataT, IndexT>(
                       centroids_raw_data.data_handle(), n_clusters, n_features),
                     itr,
                     wt_in_cluster.view(),
                     new_centroids.view(),
                     workspace);

    // compute the squared norm between the new_centroids and the original
    // centroids, destructor releases the resource
    auto sqrd_norm = raft::make_device_scalar(handle, DataT(0));
    raft::linalg::mapThenSumReduce(sqrd_norm.data_handle(),
                                   new_centroids.size(),
                                   raft::sqdiff_op{},
                                   stream,
                                   centroids.data_handle(),
                                   new_centroids.data_handle());

    DataT sqrd_norm_error = 0;
    raft::copy(&sqrd_norm_error, sqrd_norm.data_handle(), sqrd_norm.size(), stream);

    raft::copy(
      centroids_raw_data.data_handle(), new_centroids.data_handle(), new_centroids.size(), stream);

    bool done = false;
    if (params.inertia_check) {
      // calculate cluster cost phi_x(C)
      cuvs::cluster::kmeans::detail::compute_cluster_cost(
        handle,
        min_cluster_and_distance.view(),
        workspace,
        raft::make_device_scalar_view(cluster_cost_d.data()),
        raft::value_op{},
        raft::add_op{});

      DataT cur_clustering_cost = cluster_cost_d.value(stream);

      ASSERT(cur_clustering_cost != (DataT)0.0,
             "Too few points and centroids being found is getting 0 cost from "
             "centers");

      if (n_iter[0] > 1) {
        DataT delta = cur_clustering_cost / prior_clustering_cost;
        if (delta > 1 - params.tol) done = true;
      }
      prior_clustering_cost = cur_clustering_cost;
    }

    raft::resource::sync_stream(handle, stream);
    if (sqrd_norm_error < params.tol) done = true;

    if (done) {
      RAFT_LOG_DEBUG("Threshold triggered after %d iterations. Terminating early.", n_iter[0]);
      break;
    }
  }

  auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroids_raw_data.data_handle(), n_clusters, n_features);

  cuvs::cluster::kmeans::detail::min_cluster_and_distance_compute<DataT, IndexT>(
    handle,
    X,
    centroids,
    min_cluster_and_distance.view(),
    l2normx_view,
    l2_norm_buf_or_dist_buf,
    params.metric,
    params.batch_samples,
    params.batch_centroids,
    workspace);

  // TODO(snanditale): add different templates for InType of binaryOp to avoid thrust transform
  thrust::transform(raft::resource::get_thrust_policy(handle),
                    min_cluster_and_distance.data_handle(),
                    min_cluster_and_distance.data_handle() + min_cluster_and_distance.size(),
                    weight.data_handle(),
                    min_cluster_and_distance.data_handle(),
                    [=] __device__(const raft::KeyValuePair<IndexT, DataT> kvp,
                                   DataT wt) -> raft::KeyValuePair<IndexT, DataT> {
                      raft::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  // calculate cluster cost phi_x(C)
  cuvs::cluster::kmeans::detail::compute_cluster_cost(
    handle,
    min_cluster_and_distance.view(),
    workspace,
    raft::make_device_scalar_view(cluster_cost_d.data()),
    raft::value_op{},
    raft::add_op{});

  inertia[0] = cluster_cost_d.value(stream);

  RAFT_LOG_DEBUG("KMeans.fit: completed after %d iterations with %f inertia[0] ",
                 n_iter[0] > params.max_iter ? n_iter[0] - 1 : n_iter[0],
                 inertia[0]);
}

/*
 * @brief Selects 'n_clusters' samples from X using scalable kmeans++ algorithm.

 * @note  This is the algorithm described in
 *        "Scalable K-Means++", 2012, Bahman Bahmani, Benjamin Moseley,
 *         Andrea Vattani, Ravi Kumar, Sergei Vassilvitskii,
 *         https://arxiv.org/abs/1203.6402

 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: psi = phi_X (C)
 * 3: for O( log(psi) ) times do
 * 4:   C' = sample each point x in X independently with probability
 *           p_x = l * (d^2(x, C) / phi_X (C) )
 * 5:   C = C U C'
 * 6: end for
 * 7: For x in C, set w_x to be the number of points in X closer to x than any
 * other point in C
 * 8: Recluster the weighted points in C into k clusters

 * TODO: Resizing is needed to use mdarray instead of rmm::device_uvector

 */
template <typename DataT, typename IndexT>
void init_scalable_k_means_plus_plus(raft::resources const& handle,
                                     const cuvs::cluster::kmeans::params& params,
                                     raft::device_matrix_view<const DataT, IndexT> X,
                                     raft::device_matrix_view<DataT, IndexT> centroids_raw_data,
                                     rmm::device_uvector<char>& workspace)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "init_scalable_k_means_plus_plus");
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  std::mt19937 gen(params.rng_state.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  auto c_idx            = dis(gen);
  auto initial_centroid = raft::make_device_matrix_view<const DataT, IndexT>(
    X.data_handle() + c_idx * n_features, 1, n_features);

  // flag the sample that is chosen as initial centroid
  std::vector<uint8_t> h_is_sample_centroid(n_samples);
  std::fill(h_is_sample_centroid.begin(), h_is_sample_centroid.end(), 0);
  h_is_sample_centroid[c_idx] = 1;

  // device buffer to flag the sample that is chosen as initial centroid
  auto is_sample_centroid = raft::make_device_vector<uint8_t, IndexT>(handle, n_samples);

  raft::copy(is_sample_centroid.data_handle(),
             h_is_sample_centroid.data(),
             is_sample_centroid.size(),
             stream);

  rmm::device_uvector<DataT> centroids_buf(initial_centroid.size(), stream);

  // reset buffer to store the chosen centroid
  raft::copy(centroids_buf.data(), initial_centroid.data_handle(), initial_centroid.size(), stream);

  auto potential_centroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroids_buf.data(), initial_centroid.extent(0), initial_centroid.extent(1));
  // <<< End of Step-1 >>>

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> l2_norm_buf_or_dist_buf(0, stream);

  // L2 norm of X: ||x||^2
  auto l2_norm_x = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      l2_norm_x.data_handle(), X.data_handle(), X.extent(1), X.extent(0), stream);
  }

  auto min_cluster_distance_vec = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto uniform_rands            = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  rmm::device_scalar<DataT> cluster_cost(stream);

  // <<< Step-2 >>>: psi <- phi_X (C)
  cuvs::cluster::kmeans::detail::min_cluster_distance_compute<DataT, IndexT>(
    handle,
    X,
    potential_centroids,
    min_cluster_distance_vec.view(),
    l2_norm_x.view(),
    l2_norm_buf_or_dist_buf,
    params.metric,
    params.batch_samples,
    params.batch_centroids,
    workspace);

  // compute partial cluster cost from the samples in rank
  cuvs::cluster::kmeans::detail::compute_cluster_cost(
    handle,
    min_cluster_distance_vec.view(),
    workspace,
    raft::make_device_scalar_view(cluster_cost.data()),
    raft::identity_op{},
    raft::add_op{});

  auto psi = cluster_cost.value(stream);

  // <<< End of Step-2 >>>

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  raft::resource::sync_stream(handle, stream);
  int niter = std::min(8, static_cast<int>(ceil(log(psi))));
  RAFT_LOG_DEBUG("KMeans||: psi = %g, log(psi) = %g, niter = %d ", psi, log(psi), niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    RAFT_LOG_DEBUG("KMeans|| - Iteration %d: # potential centroids sampled - %d",
                   iter,
                   potential_centroids.extent(0));

    cuvs::cluster::kmeans::detail::min_cluster_distance_compute<DataT, IndexT>(
      handle,
      X,
      potential_centroids,
      min_cluster_distance_vec.view(),
      l2_norm_x.view(),
      l2_norm_buf_or_dist_buf,
      params.metric,
      params.batch_samples,
      params.batch_centroids,
      workspace);

    cuvs::cluster::kmeans::detail::compute_cluster_cost(
      handle,
      min_cluster_distance_vec.view(),
      workspace,
      raft::make_device_scalar_view<DataT>(cluster_cost.data()),
      raft::identity_op{},
      raft::add_op{});

    psi = cluster_cost.value(stream);

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potential_centroids
    raft::random::uniform(handle,
                          rng,
                          uniform_rands.data_handle(),
                          uniform_rands.extent(0),
                          static_cast<DataT>(0),
                          static_cast<DataT>(1));

    cuvs::cluster::kmeans::detail::sampling_op<DataT, IndexT> select_op(
      psi,
      params.oversampling_factor,
      n_clusters,
      uniform_rands.data_handle(),
      is_sample_centroid.data_handle());

    rmm::device_uvector<DataT> cp_raw(0, stream);
    cuvs::cluster::kmeans::detail::sample_centroids<DataT, IndexT>(handle,
                                                                   X,
                                                                   min_cluster_distance_vec.view(),
                                                                   is_sample_centroid.view(),
                                                                   select_op,
                                                                   cp_raw,
                                                                   workspace);
    auto cp = raft::make_device_matrix_view<DataT, IndexT>(
      cp_raw.data(), cp_raw.size() / n_features, n_features);
    /// <<<< End of Step-4 >>>>

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in cp to the buffer holding the potential_centroids
    centroids_buf.resize(centroids_buf.size() + cp.size(), stream);
    raft::copy(
      centroids_buf.data() + centroids_buf.size() - cp.size(), cp.data_handle(), cp.size(), stream);

    IndexT tot_centroids = potential_centroids.extent(0) + cp.extent(0);
    potential_centroids =
      raft::make_device_matrix_view<DataT, IndexT>(centroids_buf.data(), tot_centroids, n_features);
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  RAFT_LOG_DEBUG("KMeans||: total # potential centroids sampled - %d",
                 potential_centroids.extent(0));

  if (static_cast<int>(potential_centroids.extent(0)) > n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource
    auto weight = raft::make_device_vector<DataT, IndexT>(handle, potential_centroids.extent(0));

    cuvs::cluster::kmeans::detail::count_samples_in_cluster<DataT, IndexT>(
      handle, params, X, l2_norm_x.view(), potential_centroids, workspace, weight.view());

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    cuvs::cluster::kmeans::detail::kmeans_plus_plus<DataT, IndexT>(
      handle, params, potential_centroids, centroids_raw_data, workspace);

    auto inertia = raft::make_host_scalar<DataT>(0);
    auto n_iter  = raft::make_host_scalar<IndexT>(0);
    cuvs::cluster::kmeans::params default_params;
    default_params.n_clusters = params.n_clusters;

    cuvs::cluster::kmeans::detail::kmeans_fit_main<DataT, IndexT>(handle,
                                                                  default_params,
                                                                  potential_centroids,
                                                                  weight.view(),
                                                                  centroids_raw_data,
                                                                  inertia.view(),
                                                                  n_iter.view(),
                                                                  workspace);

  } else if (static_cast<int>(potential_centroids.extent(0)) < n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potential_centroids.extent(0);

    RAFT_LOG_DEBUG(
      "[Warning!] KMeans||: found fewer than %d centroids during "
      "initialization (found %d centroids, remaining %d centroids will be "
      "chosen randomly from input samples)",
      n_clusters,
      potential_centroids.extent(0),
      n_random_clusters);

    // generate `n_random_clusters` centroids
    cuvs::cluster::kmeans::params rand_params;
    rand_params.init       = cuvs::cluster::kmeans::params::InitMethod::Random;
    rand_params.n_clusters = n_random_clusters;
    init_random<DataT, IndexT>(handle, rand_params, X, centroids_raw_data);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroids_raw_data.data_handle() + n_random_clusters * n_features,
               potential_centroids.data_handle(),
               potential_centroids.size(),
               stream);
  } else {
    // found the required n_clusters
    raft::copy(centroids_raw_data.data_handle(),
               potential_centroids.data_handle(),
               potential_centroids.size(),
               stream);
  }
}

/**
 * @brief Find clusters with k-means algorithm.
 *   Initial centroids are chosen with k-means++ algorithm. Empty
 *   clusters are reinitialized by choosing new centroids with
 *   k-means++ algorithm.
 * @tparam DataT the type of data used for weights, distances.
 * @tparam IndexT the type of data used for indexing.
 * @param[in]     handle        The raft handle.
 * @param[in]     params        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. It must be noted
 * that the data must be in row-major format and stored in device accessible
 * location.
 * @param[in]     n_samples     Number of samples in the input X.
 * @param[in]     n_features    Number of features or the dimensions of each
 * sample.
 * @param[in]     sample_weight Optional weights for each observation in X.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, use
 * centroids as the initial cluster centers
 *                              [out] Otherwise, generated centroids from the
 * kmeans algorithm is stored at the address pointed by 'centroids'.
 * @param[out]    inertia       Sum of squared distances of samples to their
 * closest cluster center.
 * @param[out]    n_iter        Number of iterations run.
 */
template <typename DataT, typename IndexT>
void kmeans_fit(raft::resources const& handle,
                const cuvs::cluster::kmeans::params& pams,
                raft::device_matrix_view<const DataT, IndexT> X,
                std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                raft::device_matrix_view<DataT, IndexT> centroids,
                raft::host_scalar_view<DataT> inertia,
                raft::host_scalar_view<IndexT> n_iter)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeans_fit");
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = pams.n_clusters;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // Check that parameters are valid
  if (sample_weight.has_value()) {
    RAFT_EXPECTS(sample_weight.value().extent(0) == n_samples,
                 "invalid parameter (sample_weight!=n_samples)");
  }
  RAFT_EXPECTS(n_clusters > 0, "invalid parameter (n_clusters<=0)");
  RAFT_EXPECTS(pams.tol > 0, "invalid parameter (tol<=0)");
  RAFT_EXPECTS(pams.oversampling_factor >= 0, "invalid parameter (oversampling_factor<0)");
  RAFT_EXPECTS((int)centroids.extent(0) == pams.n_clusters,
               "invalid parameter (centroids.extent(0) != n_clusters)");
  RAFT_EXPECTS(centroids.extent(1) == n_features,
               "invalid parameter (centroids.extent(1) != n_features)");

  // Display a message if the batch size is smaller than n_samples but will be ignored
  if (pams.batch_samples < static_cast<int>(n_samples) &&
      (pams.metric == cuvs::distance::DistanceType::L2Expanded ||
       pams.metric == cuvs::distance::DistanceType::L2SqrtExpanded)) {
    RAFT_LOG_DEBUG(
      "batch_samples=%d was passed, but batch_samples=%d will be used (reason: "
      "batch_samples has no impact on the memory footprint when FusedL2NN can be used)",
      pams.batch_samples,
      (int)n_samples);
  }
  // Display a message if batch_centroids is set and a fusedL2NN-compatible metric is used
  if (pams.batch_centroids != 0 && pams.batch_centroids != pams.n_clusters &&
      (pams.metric == cuvs::distance::DistanceType::L2Expanded ||
       pams.metric == cuvs::distance::DistanceType::L2SqrtExpanded)) {
    RAFT_LOG_DEBUG(
      "batch_centroids=%d was passed, but batch_centroids=%d will be used (reason: "
      "batch_centroids has no impact on the memory footprint when FusedL2NN can be used)",
      pams.batch_centroids,
      pams.n_clusters);
  }

  raft::default_logger().set_level(pams.verbosity);

  // Allocate memory
  rmm::device_uvector<char> workspace(0, stream);
  auto weight = raft::make_device_vector<DataT>(handle, n_samples);
  if (sample_weight.has_value()) {
    raft::copy(weight.data_handle(), sample_weight.value().data_handle(), n_samples, stream);
  } else {
    thrust::fill(raft::resource::get_thrust_policy(handle),
                 weight.data_handle(),
                 weight.data_handle() + weight.size(),
                 1);
  }

  // check if weights sum up to n_samples
  check_weight<DataT>(handle, weight.view(), workspace);

  auto centroids_raw_data = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  auto n_init = pams.n_init;
  if (pams.init == cuvs::cluster::kmeans::params::InitMethod::Array && n_init != 1) {
    RAFT_LOG_DEBUG(
      "Explicit initial center position passed: performing only one init in "
      "k-means instead of n_init=%d",
      n_init);
    n_init = 1;
  }

  std::mt19937 gen(pams.rng_state.seed);
  inertia[0] = std::numeric_limits<DataT>::max();

  for (auto seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    cuvs::cluster::kmeans::params iter_params = pams;
    iter_params.rng_state.seed                = gen();

    DataT iter_inertia    = std::numeric_limits<DataT>::max();
    IndexT n_current_iter = 0;
    if (iter_params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
      // initializing with random samples from input dataset
      RAFT_LOG_DEBUG(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers by "
        "randomly choosing from the "
        "input data.",
        seed_iter + 1,
        n_init);
      init_random<DataT, IndexT>(handle, iter_params, X, centroids_raw_data.view());
    } else if (iter_params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
      // default method to initialize is kmeans++
      RAFT_LOG_DEBUG(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers using "
        "k-means++ algorithm.",
        seed_iter + 1,
        n_init);
      if (iter_params.oversampling_factor == 0) {
        cuvs::cluster::kmeans::detail::kmeans_plus_plus<DataT, IndexT>(
          handle, iter_params, X, centroids_raw_data.view(), workspace);
      } else {
        cuvs::cluster::kmeans::detail::init_scalable_k_means_plus_plus<DataT, IndexT>(
          handle, iter_params, X, centroids_raw_data.view(), workspace);
      }
    } else if (iter_params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
      RAFT_LOG_DEBUG(
        "KMeans.fit (Iteration-%d/%d): initialize cluster centers from "
        "the ndarray array input "
        "passed to init argument.",
        seed_iter + 1,
        n_init);
      raft::copy(
        centroids_raw_data.data_handle(), centroids.data_handle(), n_clusters * n_features, stream);
    } else {
      THROW("unknown initialization method to select initial centers");
    }

    cuvs::cluster::kmeans::detail::kmeans_fit_main<DataT, IndexT>(
      handle,
      iter_params,
      X,
      weight.view(),
      centroids_raw_data.view(),
      raft::make_host_scalar_view<DataT>(&iter_inertia),
      raft::make_host_scalar_view<IndexT>(&n_current_iter),
      workspace);
    if (iter_inertia < inertia[0]) {
      inertia[0] = iter_inertia;
      n_iter[0]  = n_current_iter;
      raft::copy(
        centroids.data_handle(), centroids_raw_data.data_handle(), n_clusters * n_features, stream);
    }
    RAFT_LOG_DEBUG("KMeans.fit after iteration-%d/%d: inertia - %f, n_iter[0] - %d",
                   seed_iter + 1,
                   n_init,
                   inertia[0],
                   n_iter[0]);
  }
  RAFT_LOG_DEBUG("KMeans.fit: async call returned (fit could still be running on the device)");
}

template <typename DataT, typename IndexT = int>
void kmeans_fit(raft::resources const& handle,
                const cuvs::cluster::kmeans::params& pams,
                const DataT* X,
                const DataT* sample_weight,
                DataT* centroids,
                IndexT n_samples,
                IndexT n_features,
                DataT& inertia,
                IndexT& n_iter)
{
  auto x_view = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<DataT, IndexT>(centroids, pams.n_clusters, n_features);
  std::optional<raft::device_vector_view<const DataT>> sample_weightView = std::nullopt;
  if (sample_weight)
    sample_weightView =
      raft::make_device_vector_view<const DataT, IndexT>(sample_weight, n_samples);
  auto inertia_view = raft::make_host_scalar_view(&inertia);
  auto n_iter_view  = raft::make_host_scalar_view(&n_iter);

  cuvs::cluster::kmeans::detail::kmeans_fit<DataT, IndexT>(
    handle, pams, x_view, sample_weightView, centroids_view, inertia_view, n_iter_view);
}

template <typename DataT, typename IndexT>
void kmeans_predict(raft::resources const& handle,
                    const cuvs::cluster::kmeans::params& pams,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
                    raft::device_matrix_view<const DataT, IndexT> centroids,
                    raft::device_vector_view<IndexT, IndexT> labels,
                    bool normalize_weight,
                    raft::host_scalar_view<DataT> inertia)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeans_predict");
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // Check that parameters are valid
  if (sample_weight.has_value()) {
    RAFT_EXPECTS(sample_weight.value().extent(0) == n_samples,
                 "invalid parameter (sample_weight!=n_samples)");
  }
  RAFT_EXPECTS(pams.n_clusters > 0, "invalid parameter (n_clusters<=0)");
  RAFT_EXPECTS(pams.tol > 0, "invalid parameter (tol<=0)");
  RAFT_EXPECTS(pams.oversampling_factor >= 0, "invalid parameter (oversampling_factor<0)");
  RAFT_EXPECTS((int)centroids.extent(0) == pams.n_clusters,
               "invalid parameter (centroids.extent(0) != n_clusters)");
  RAFT_EXPECTS(centroids.extent(1) == n_features,
               "invalid parameter (centroids.extent(1) != n_features)");

  raft::default_logger().set_level(pams.verbosity);
  auto metric = pams.metric;

  // Allocate memory
  // Device-accessible allocation of expandable storage used as temporary buffers
  rmm::device_uvector<char> workspace(0, stream);
  auto weight = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (sample_weight.has_value()) {
    raft::copy(weight.data_handle(), sample_weight.value().data_handle(), n_samples, stream);
  } else {
    thrust::fill(raft::resource::get_thrust_policy(handle),
                 weight.data_handle(),
                 weight.data_handle() + weight.size(),
                 1);
  }

  // check if weights sum up to n_samples
  if (normalize_weight) check_weight(handle, weight.view(), workspace);

  auto min_cluster_and_distance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
  rmm::device_uvector<DataT> l2_norm_buf_or_dist_buf(0, stream);

  // L2 norm of X: ||x||^2
  auto l2_norm_x = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      l2_norm_x.data_handle(), X.data_handle(), X.extent(1), X.extent(0), stream);
  }

  // computes min_cluster_and_distance[0:n_samples) where  min_cluster_and_distance[i]
  // is a <key, value> pair where
  //   'key' is index to a sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  auto l2normx_view =
    raft::make_device_vector_view<const DataT, IndexT>(l2_norm_x.data_handle(), n_samples);
  cuvs::cluster::kmeans::detail::min_cluster_and_distance_compute<DataT, IndexT>(
    handle,
    X,
    centroids,
    min_cluster_and_distance.view(),
    l2normx_view,
    l2_norm_buf_or_dist_buf,
    pams.metric,
    pams.batch_samples,
    pams.batch_centroids,
    workspace);

  // calculate cluster cost phi_x(C)
  rmm::device_scalar<DataT> cluster_cost_d(stream);
  // TODO(snanditale): add different templates for InType of binaryOp to avoid thrust transform
  thrust::transform(raft::resource::get_thrust_policy(handle),
                    min_cluster_and_distance.data_handle(),
                    min_cluster_and_distance.data_handle() + min_cluster_and_distance.size(),
                    weight.data_handle(),
                    min_cluster_and_distance.data_handle(),
                    [=] __device__(const raft::KeyValuePair<IndexT, DataT> kvp,
                                   DataT wt) -> raft::KeyValuePair<IndexT, DataT> {
                      raft::KeyValuePair<IndexT, DataT> res;
                      res.value = kvp.value * wt;
                      res.key   = kvp.key;
                      return res;
                    });

  cuvs::cluster::kmeans::detail::compute_cluster_cost(
    handle,
    min_cluster_and_distance.view(),
    workspace,
    raft::make_device_scalar_view(cluster_cost_d.data()),
    raft::value_op{},
    raft::add_op{});

  thrust::transform(raft::resource::get_thrust_policy(handle),
                    min_cluster_and_distance.data_handle(),
                    min_cluster_and_distance.data_handle() + min_cluster_and_distance.size(),
                    labels.data_handle(),
                    raft::key_op{});

  inertia[0] = cluster_cost_d.value(stream);
}

template <typename DataT, typename IndexT = int>
void kmeans_predict(raft::resources const& handle,
                    const cuvs::cluster::kmeans::params& pams,
                    const DataT* X,
                    const DataT* sample_weight,
                    const DataT* centroids,
                    IndexT n_samples,
                    IndexT n_features,
                    IndexT* labels,
                    bool normalize_weight,
                    DataT& inertia)
{
  auto x_view = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<const DataT, IndexT>(centroids, pams.n_clusters, n_features);
  std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weightView{std::nullopt};
  if (sample_weight)
    sample_weightView.emplace(
      raft::make_device_vector_view<const DataT, IndexT>(sample_weight, n_samples));
  auto labels_view  = raft::make_device_vector_view<IndexT, IndexT>(labels, n_samples);
  auto inertia_view = raft::make_host_scalar_view(&inertia);

  cuvs::cluster::kmeans::detail::kmeans_predict<DataT, IndexT>(handle,
                                                               pams,
                                                               x_view,
                                                               sample_weightView,
                                                               centroids_view,
                                                               labels_view,
                                                               normalize_weight,
                                                               inertia_view);
}

/**
 * @brief Transform X to a cluster-distance space.
 *
 * @param[in]     handle        The handle to the cuML library context that
 * manages the CUDA resources.
 * @param[in]     pams        Parameters for KMeans model.
 * @param[in]     X             Training instances to cluster. The data must
 * be in row-major format
 * @param[in]     centroids     Cluster centroids. The data must be in row-major format.
 * @param[out]    X_new         X transformed in the new space..
 */
template <typename DataT, typename IndexT = int>
void kmeans_transform(raft::resources const& handle,
                      const cuvs::cluster::kmeans::params& pams,
                      raft::device_matrix_view<const DataT, IndexT> X,
                      raft::device_matrix_view<const DataT, IndexT> centroids,
                      raft::device_matrix_view<DataT, IndexT> X_new)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeans_transform");
  raft::default_logger().set_level(pams.verbosity);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = pams.n_clusters;
  auto metric         = pams.metric;

  // Device-accessible allocation of expandable storage used as temporary buffers
  rmm::device_uvector<char> workspace(0, stream);
  auto data_batch_size = get_data_batch_size(pams.batch_samples, n_samples);

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (IndexT d_idx = 0; d_idx < static_cast<IndexT>(n_samples); d_idx += data_batch_size) {
    // # of samples for the current batch
    auto ns =
      std::min(static_cast<IndexT>(data_batch_size), static_cast<IndexT>(n_samples - d_idx));

    // dataset_view [ns x n_features] - view representing the current batch of
    // input dataset
    auto dataset_view = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + n_features * d_idx, ns, n_features);

    // pairwise_distance_view [ns x n_clusters]
    auto pairwise_distance_view = raft::make_device_matrix_view<DataT, IndexT>(
      X_new.data_handle() + n_clusters * d_idx, ns, n_clusters);

    // calculate pairwise distance between cluster centroids and current batch
    // of input dataset
    pairwise_distance_kmeans<DataT, IndexT>(
      handle, dataset_view, centroids, pairwise_distance_view, metric);
  }
}
}  // namespace cuvs::cluster::kmeans::detail
