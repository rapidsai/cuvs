/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../core/nvtx.hpp"
#include "../../neighbors/detail/ann_utils.cuh"
#include "kmeans_common.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <optional>
#include <random>

namespace cuvs::cluster::kmeans::detail {

static const std::string CUVS_NAME = "cuvs";

// =========================================================
// Init functions
// =========================================================

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(raft::resources const& handle,
                const cuvs::cluster::kmeans::params& params,
                raft::device_matrix_view<const DataT, IndexT> X,
                raft::device_matrix_view<DataT, IndexT> centroids)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("initRandom");
  auto n_clusters = params.n_clusters;
  cuvs::cluster::kmeans::detail::shuffleAndGather<DataT, IndexT>(
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
void kmeansPlusPlus(raft::resources const& handle,
                    const cuvs::cluster::kmeans::params& params,
                    raft::device_matrix_view<const DataT, IndexT> X,
                    raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                    rmm::device_uvector<char>& workspace)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeansPlusPlus");
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

  auto dataBatchSize = getDataBatchSize(params.batch_samples, n_samples);

  // temporary buffers
  auto indices            = raft::make_device_vector<IndexT, IndexT>(handle, n_trials);
  auto centroidCandidates = raft::make_device_matrix<DataT, IndexT>(handle, n_trials, n_features);
  auto costPerCandidate   = raft::make_device_vector<DataT, IndexT>(handle, n_trials);
  auto minClusterDistance = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto distBuffer         = raft::make_device_matrix<DataT, IndexT>(handle, n_trials, n_samples);

  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_scalar<DataT> clusterCost(stream);
  rmm::device_scalar<cub::KeyValuePair<int, DataT>> minClusterIndexAndDistance(stream);

  // Device and matrix views
  raft::device_vector_view<IndexT, IndexT> indices_view(indices.data_handle(), n_trials);
  auto const_weights_view =
    raft::make_device_vector_view<const DataT, IndexT>(minClusterDistance.data_handle(), n_samples);
  auto const_indices_view =
    raft::make_device_vector_view<const IndexT, IndexT>(indices.data_handle(), n_trials);
  auto const_X_view =
    raft::make_device_matrix_view<const DataT, IndexT>(X.data_handle(), n_samples, n_features);
  raft::device_matrix_view<DataT, IndexT> candidates_view(
    centroidCandidates.data_handle(), n_trials, n_features);

  // L2 norm of X: ||c||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);

  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle, X, L2NormX.view());
  }

  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  std::mt19937 gen(params.rng_state.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  // <<< Step-1 >>>: C <-- sample a point uniformly at random from X
  auto initialCentroid = raft::make_device_matrix_view<const DataT, IndexT>(
    X.data_handle() + dis(gen) * n_features, 1, n_features);
  int n_clusters_picked = 1;

  // store the chosen centroid in the buffer
  raft::copy(handle,
             raft::make_device_vector_view(centroidsRawData.data_handle(), initialCentroid.size()),
             raft::make_device_vector_view(initialCentroid.data_handle(), initialCentroid.size()));

  //  C = initial set of centroids
  auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroidsRawData.data_handle(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  // Calculate cluster distance, d^2(x, C), for all the points x in X to the nearest centroid
  cuvs::cluster::kmeans::detail::minClusterDistanceCompute<DataT, IndexT>(handle,
                                                                          X,
                                                                          centroids,
                                                                          minClusterDistance.view(),
                                                                          L2NormX.view(),
                                                                          L2NormBuf_OR_DistBuf,
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
    raft::matrix::gather(handle, const_X_view, const_indices_view, candidates_view);

    // Calculate pairwise distance between X and the centroid candidates
    // Output - pwd [n_trials x n_samples]
    auto pwd = distBuffer.view();
    cuvs::cluster::kmeans::detail::pairwise_distance_kmeans<DataT, IndexT>(
      handle, centroidCandidates.view(), X, pwd, metric);

    // Update nearest cluster distance for each centroid candidate
    // Note pwd and minDistBuf points to same buffer which currently holds pairwise distance values.
    // Outputs minDistanceBuf[n_trials x n_samples] where minDistance[i, :] contains updated
    // minClusterDistance that includes candidate-i
    auto minDistBuf = distBuffer.view();
    raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_const_mdspan(pwd),
      raft::make_const_mdspan(minClusterDistance.view()),
      minDistBuf,
      raft::min_op{});

    // Calculate costPerCandidate[n_trials] where costPerCandidate[i] is the cluster cost when using
    // centroid candidate-i
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<const DataT, IndexT, raft::row_major>(
        minDistBuf.data_handle(), minDistBuf.extent(0), minDistBuf.extent(1)),
      raft::make_device_vector_view<DataT, IndexT>(costPerCandidate.data_handle(),
                                                   minDistBuf.extent(0)),
      static_cast<DataT>(0));

    // Greedy Choice - Choose the candidate that has minimum cluster cost
    // ArgMin operation below identifies the index of minimum cost in costPerCandidate
    {
      // Determine temporary device storage requirements
      size_t temp_storage_bytes = 0;
      cub::DeviceReduce::ArgMin(nullptr,
                                temp_storage_bytes,
                                costPerCandidate.data_handle(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.extent(0),
                                stream);

      // Allocate temporary storage
      workspace.resize(temp_storage_bytes, stream);

      // Run argmin-reduction
      cub::DeviceReduce::ArgMin(workspace.data(),
                                temp_storage_bytes,
                                costPerCandidate.data_handle(),
                                minClusterIndexAndDistance.data(),
                                costPerCandidate.extent(0),
                                stream);

      int bestCandidateIdx = -1;
      raft::copy(handle,
                 raft::make_host_scalar_view(&bestCandidateIdx),
                 raft::make_device_scalar_view(&minClusterIndexAndDistance.data()->key));
      raft::resource::sync_stream(handle);
      /// <<< End of Step-3 >>>

      /// <<< Step-4 >>>: C = C U {x}
      // Update minimum cluster distance corresponding to the chosen centroid candidate
      raft::copy(handle,
                 raft::make_device_vector_view(minClusterDistance.data_handle(), n_samples),
                 raft::make_device_vector_view(
                   minDistBuf.data_handle() + bestCandidateIdx * n_samples, n_samples));

      raft::copy(handle,
                 raft::make_device_vector_view(
                   centroidsRawData.data_handle() + n_clusters_picked * n_features, n_features),
                 raft::make_device_vector_view(
                   centroidCandidates.data_handle() + bestCandidateIdx * n_features, n_features));

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

                      // TODO: Figure out how to best wrap iterator types in mdspan
                      LabelsIterator cluster_labels,
                      raft::device_vector_view<DataT, IndexT> weight_per_cluster,
                      raft::device_matrix_view<DataT, IndexT, raft::row_major> new_centroids,
                      rmm::device_uvector<char>& workspace)
{
  auto n_clusters = centroids.extent(0);

  cuvs::cluster::kmeans::detail::compute_centroid_adjustments(handle,
                                                              X,
                                                              sample_weights,
                                                              cluster_labels,
                                                              static_cast<IndexT>(n_clusters),
                                                              new_centroids,
                                                              weight_per_cluster,
                                                              workspace);

  cuvs::cluster::kmeans::detail::finalize_centroids(handle,
                                                    raft::make_const_mdspan(new_centroids),
                                                    raft::make_const_mdspan(weight_per_cluster),
                                                    centroids,
                                                    new_centroids);
}

template <typename DataT, typename IndexT, typename Accessor>
void kmeans_fit(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& pams,
  raft::mdspan<const DataT, raft::matrix_extent<IndexT>, raft::row_major, Accessor> X,
  std::optional<
    raft::mdspan<const DataT, raft::vector_extent<IndexT>, raft::layout_right, Accessor>>
    sample_weight,
  raft::device_matrix_view<DataT, IndexT> centroids,
  raft::host_scalar_view<DataT> inertia,
  raft::host_scalar_view<IndexT> n_iter);

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
void initScalableKMeansPlusPlus(raft::resources const& handle,
                                const cuvs::cluster::kmeans::params& params,
                                raft::device_matrix_view<const DataT, IndexT> X,
                                raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                                rmm::device_uvector<char>& workspace)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "initScalableKMeansPlusPlus");
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  std::mt19937 gen(params.rng_state.seed);
  std::uniform_int_distribution<> dis(0, n_samples - 1);

  auto cIdx            = dis(gen);
  auto initialCentroid = raft::make_device_matrix_view<const DataT, IndexT>(
    X.data_handle() + cIdx * n_features, 1, n_features);

  // flag the sample that is chosen as initial centroid
  std::vector<uint8_t> h_isSampleCentroid(n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);
  h_isSampleCentroid[cIdx] = 1;

  // device buffer to flag the sample that is chosen as initial centroid
  auto isSampleCentroid = raft::make_device_vector<uint8_t, IndexT>(handle, n_samples);

  raft::copy(handle,
             raft::make_device_vector_view(isSampleCentroid.data_handle(), isSampleCentroid.size()),
             raft::make_host_vector_view(h_isSampleCentroid.data(), isSampleCentroid.size()));

  rmm::device_uvector<DataT> centroidsBuf(initialCentroid.size(), stream);

  // reset buffer to store the chosen centroid
  raft::copy(handle,
             raft::make_device_vector_view(centroidsBuf.data(), initialCentroid.size()),
             raft::make_device_vector_view(initialCentroid.data_handle(), initialCentroid.size()));

  auto potentialCentroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroidsBuf.data(), initialCentroid.extent(0), initialCentroid.extent(1));
  // <<< End of Step-1 >>>

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle, X, L2NormX.view());
  }

  auto minClusterDistanceVec = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto uniformRands          = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  rmm::device_scalar<DataT> clusterCost(stream);

  // <<< Step-2 >>>: psi <- phi_X (C)
  cuvs::cluster::kmeans::detail::minClusterDistanceCompute<DataT, IndexT>(
    handle,
    X,
    potentialCentroids,
    minClusterDistanceVec.view(),
    L2NormX.view(),
    L2NormBuf_OR_DistBuf,
    params.metric,
    params.batch_samples,
    params.batch_centroids,
    workspace);

  // compute partial cluster cost from the samples in rank
  cuvs::cluster::kmeans::detail::computeClusterCost(
    handle,
    minClusterDistanceVec.view(),
    workspace,
    raft::make_device_scalar_view(clusterCost.data()),
    raft::identity_op{},
    raft::add_op{});

  auto psi = clusterCost.value(stream);

  // <<< End of Step-2 >>>

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  raft::resource::sync_stream(handle, stream);
  int niter = std::min(8, (int)ceil(log(psi)));
  RAFT_LOG_DEBUG("KMeans||: psi = %g, log(psi) = %g, niter = %d ", psi, log(psi), niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    RAFT_LOG_DEBUG("KMeans|| - Iteration %d: # potential centroids sampled - %d",
                   iter,
                   potentialCentroids.extent(0));

    cuvs::cluster::kmeans::detail::minClusterDistanceCompute<DataT, IndexT>(
      handle,
      X,
      potentialCentroids,
      minClusterDistanceVec.view(),
      L2NormX.view(),
      L2NormBuf_OR_DistBuf,
      params.metric,
      params.batch_samples,
      params.batch_centroids,
      workspace);

    cuvs::cluster::kmeans::detail::computeClusterCost(
      handle,
      minClusterDistanceVec.view(),
      workspace,
      raft::make_device_scalar_view<DataT>(clusterCost.data()),
      raft::identity_op{},
      raft::add_op{});

    psi = clusterCost.value(stream);

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potentialCentroids
    raft::random::uniform(
      handle, rng, uniformRands.data_handle(), uniformRands.extent(0), (DataT)0, (DataT)1);

    cuvs::cluster::kmeans::detail::SamplingOp<DataT, IndexT> select_op(
      psi,
      params.oversampling_factor,
      n_clusters,
      uniformRands.data_handle(),
      isSampleCentroid.data_handle());

    rmm::device_uvector<DataT> CpRaw(0, stream);
    cuvs::cluster::kmeans::detail::sampleCentroids<DataT, IndexT>(handle,
                                                                  X,
                                                                  minClusterDistanceVec.view(),
                                                                  isSampleCentroid.view(),
                                                                  select_op,
                                                                  CpRaw,
                                                                  workspace);
    auto Cp = raft::make_device_matrix_view<DataT, IndexT>(
      CpRaw.data(), CpRaw.size() / n_features, n_features);
    /// <<<< End of Step-4 >>>>

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp to the buffer holding the potentialCentroids
    centroidsBuf.resize(centroidsBuf.size() + Cp.size(), stream);
    raft::copy(handle,
               raft::make_device_vector_view(centroidsBuf.data() + centroidsBuf.size() - Cp.size(),
                                             Cp.size()),
               raft::make_device_vector_view(Cp.data_handle(), Cp.size()));

    IndexT tot_centroids = potentialCentroids.extent(0) + Cp.extent(0);
    potentialCentroids =
      raft::make_device_matrix_view<DataT, IndexT>(centroidsBuf.data(), tot_centroids, n_features);
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  RAFT_LOG_DEBUG("KMeans||: total # potential centroids sampled - %d",
                 potentialCentroids.extent(0));

  if ((int)potentialCentroids.extent(0) > n_clusters) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource
    auto weight = raft::make_device_vector<DataT, IndexT>(handle, potentialCentroids.extent(0));

    cuvs::cluster::kmeans::detail::countSamplesInCluster<DataT, IndexT>(
      handle, params, X, L2NormX.view(), potentialCentroids, workspace, weight.view());

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    cuvs::cluster::kmeans::detail::kmeansPlusPlus<DataT, IndexT>(
      handle, params, potentialCentroids, centroidsRawData, workspace);

    auto inertia = raft::make_host_scalar<DataT>(0);
    auto n_iter  = raft::make_host_scalar<IndexT>(0);
    cuvs::cluster::kmeans::params recluster_params;
    recluster_params.n_clusters = params.n_clusters;
    recluster_params.init       = cuvs::cluster::kmeans::params::InitMethod::Array;
    recluster_params.n_init     = 1;

    auto weight_opt = std::make_optional(raft::make_const_mdspan(weight.view()));
    cuvs::cluster::kmeans::detail::kmeans_fit<DataT, IndexT>(
      handle,
      recluster_params,
      raft::make_const_mdspan(potentialCentroids),
      weight_opt,
      centroidsRawData,
      inertia.view(),
      n_iter.view());

  } else if ((int)potentialCentroids.extent(0) < n_clusters) {
    // supplement with random
    auto n_random_clusters = n_clusters - potentialCentroids.extent(0);

    RAFT_LOG_DEBUG(
      "[Warning!] KMeans||: found fewer than %d centroids during "
      "initialization (found %d centroids, remaining %d centroids will be "
      "chosen randomly from input samples)",
      n_clusters,
      potentialCentroids.extent(0),
      n_random_clusters);

    // generate `n_random_clusters` centroids
    cuvs::cluster::kmeans::params rand_params;
    rand_params.init       = cuvs::cluster::kmeans::params::InitMethod::Random;
    rand_params.n_clusters = n_random_clusters;
    initRandom<DataT, IndexT>(handle, rand_params, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(
      handle,
      raft::make_device_vector_view(centroidsRawData.data_handle() + n_random_clusters * n_features,
                                    potentialCentroids.size()),
      raft::make_device_vector_view(potentialCentroids.data_handle(), potentialCentroids.size()));
  } else {
    // found the required n_clusters
    raft::copy(
      handle,
      raft::make_device_vector_view(centroidsRawData.data_handle(), potentialCentroids.size()),
      raft::make_device_vector_view(potentialCentroids.data_handle(), potentialCentroids.size()));
  }
}

/**
 * @brief Unified k-means fit (works with host or device data).
 *
 * @tparam DataT    Data / weight type
 * @tparam IndexT   Index type
 * @tparam Accessor Accessor policy (host or device); deduced from X
 *
 * @param[in]     handle        The raft handle.
 * @param[in]     pams          Parameters for the KMeans model.
 * @param[in]     X             Training instances to cluster (host or device).
 *                              Row-major, [n_samples x n_features].
 * @param[in]     sample_weight Optional weights for each observation in X.
 *                              [n_samples].  When std::nullopt, uniform weights
 *                              are used.
 * @param[inout]  centroids     [in] When init is InitMethod::Array, used as
 *                              the initial cluster centers.
 *                              [out] The final centroids produced by the
 *                              algorithm.  [n_clusters x n_features].
 * @param[out]    inertia       Sum of squared distances of samples to their
 *                              closest cluster center.
 * @param[out]    n_iter        Number of iterations run for the best
 *                              initialization.
 */
template <typename DataT, typename IndexT, typename Accessor>
void kmeans_fit(
  raft::resources const& handle,
  const cuvs::cluster::kmeans::params& pams,
  raft::mdspan<const DataT, raft::matrix_extent<IndexT>, raft::row_major, Accessor> X,
  std::optional<
    raft::mdspan<const DataT, raft::vector_extent<IndexT>, raft::layout_right, Accessor>>
    sample_weight,
  raft::device_matrix_view<DataT, IndexT> centroids,
  raft::host_scalar_view<DataT> inertia,
  raft::host_scalar_view<IndexT> n_iter)
{
  raft::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope("kmeans_fit");
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = pams.n_clusters;
  auto metric         = pams.metric;
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  if (sample_weight.has_value())
    RAFT_EXPECTS(sample_weight.value().extent(0) == n_samples,
                 "invalid parameter (sample_weight!=n_samples)");
  RAFT_EXPECTS(n_clusters > 0, "invalid parameter (n_clusters<=0)");
  RAFT_EXPECTS(pams.tol > 0, "invalid parameter (tol<=0)");
  RAFT_EXPECTS(pams.oversampling_factor >= 0, "invalid parameter (oversampling_factor<0)");
  RAFT_EXPECTS(static_cast<IndexT>(centroids.extent(0)) == n_clusters,
               "invalid parameter (centroids.extent(0) != n_clusters)");
  RAFT_EXPECTS(centroids.extent(1) == n_features,
               "invalid parameter (centroids.extent(1) != n_features)");

  raft::default_logger().set_level(pams.verbosity);

  IndexT streaming_batch_size = static_cast<IndexT>(pams.streaming_batch_size);
  if (streaming_batch_size <= 0 || streaming_batch_size > static_cast<IndexT>(n_samples)) {
    streaming_batch_size = static_cast<IndexT>(n_samples);
  }

  const DataT* weight_ptr =
    sample_weight.has_value() ? sample_weight.value().data_handle() : nullptr;
  DataT weight_scale = compute_weight_scale<DataT, IndexT>(weight_ptr, n_samples, stream);

  rmm::device_uvector<char> workspace(0, stream);

  constexpr bool data_on_device = !raft::is_host_mdspan_v<decltype(X)>;

  auto init_centroids = [&](const cuvs::cluster::kmeans::params& iter_params,
                            raft::device_matrix_view<DataT, IndexT> centroidsRawData) {
    if (iter_params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
      raft::copy(
        handle,
        raft::make_device_vector_view(centroidsRawData.data_handle(), n_clusters * n_features),
        raft::make_device_vector_view(centroids.data_handle(), n_clusters * n_features));
      return;
    }

    raft::random::RngState random_state(iter_params.rng_state.seed);

    if (iter_params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
      raft::matrix::sample_rows(handle, random_state, X, centroidsRawData);
    } else if (iter_params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
      IndexT default_init_size =
        data_on_device ? n_samples : std::min(static_cast<IndexT>(3 * n_clusters), n_samples);
      IndexT init_sample_size = iter_params.init_size > 0
                                  ? std::min(static_cast<IndexT>(iter_params.init_size), n_samples)
                                  : default_init_size;

      auto init_sample =
        raft::make_device_matrix<DataT, IndexT>(handle, init_sample_size, n_features);
      raft::matrix::sample_rows(handle, random_state, X, init_sample.view());

      auto init_sample_const = raft::make_const_mdspan(init_sample.view());
      if (iter_params.oversampling_factor == 0)
        kmeansPlusPlus<DataT, IndexT>(
          handle, iter_params, init_sample_const, centroidsRawData, workspace);
      else
        initScalableKMeansPlusPlus<DataT, IndexT>(
          handle, iter_params, init_sample_const, centroidsRawData, workspace);
    } else {
      THROW("unknown initialization method to select initial centers");
    }
  };

  auto n_init = pams.n_init;
  if (pams.init == cuvs::cluster::kmeans::params::InitMethod::Array && n_init != 1) {
    RAFT_LOG_DEBUG(
      "Explicit initial center position passed: performing only one init in "
      "k-means instead of n_init=%d",
      n_init);
    n_init = 1;
  }

  auto centroidsRawData = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

  auto minClusterAndDistance = raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(
    handle, streaming_batch_size);
  auto L2NormBatch       = raft::make_device_vector<DataT, IndexT>(handle, streaming_batch_size);
  auto batch_weights_buf = raft::make_device_vector<DataT, IndexT>(handle, streaming_batch_size);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  auto centroid_sums      = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto weight_per_cluster = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);
  auto new_centroids      = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto clustering_cost    = raft::make_device_scalar<DataT>(handle, DataT{0});
  auto batch_sums         = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto batch_counts       = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);

  cuvs::spatial::knn::detail::utils::batch_load_iterator<DataT> data_batches(
    X.data_handle(), n_samples, n_features, streaming_batch_size, stream);
  cuvs::spatial::knn::detail::utils::batch_load_iterator<DataT> weight_batches(
    weight_ptr, n_samples, 1, streaming_batch_size, stream);

  if (weight_ptr == nullptr) { raft::matrix::fill(handle, batch_weights_buf.view(), DataT{1}); }

  auto prepare_batch_weights = [&](const auto& wt_batch, IndexT cur_batch_size) {
    if (weight_ptr != nullptr) {
      raft::copy(batch_weights_buf.data_handle(), wt_batch.data(), cur_batch_size, stream);
      if (weight_scale != DataT{1}) {
        auto bw = raft::make_device_vector_view<DataT, IndexT>(batch_weights_buf.data_handle(),
                                                               cur_batch_size);
        raft::linalg::map(
          handle, bw, raft::mul_const_op<DataT>{weight_scale}, raft::make_const_mdspan(bw));
      }
    }
    return raft::make_device_vector_view<const DataT, IndexT>(batch_weights_buf.data_handle(),
                                                              cur_batch_size);
  };

  RAFT_LOG_DEBUG(
    "KMeans.fit: n_samples=%zu, n_features=%zu, n_clusters=%d, streaming_batch_size=%zu",
    static_cast<size_t>(n_samples),
    static_cast<size_t>(n_features),
    n_clusters,
    static_cast<size_t>(streaming_batch_size));

  std::mt19937 gen(pams.rng_state.seed);
  inertia[0] = std::numeric_limits<DataT>::max();

  for (int seed_iter = 0; seed_iter < n_init; ++seed_iter) {
    cuvs::cluster::kmeans::params iter_params = pams;
    iter_params.rng_state.seed                = gen();

    RAFT_LOG_DEBUG("KMeans.fit: n_init iteration %d/%d (seed=%llu)",
                   seed_iter + 1,
                   n_init,
                   (unsigned long long)iter_params.rng_state.seed);

    init_centroids(iter_params, centroidsRawData.view());

    DataT iter_inertia        = std::numeric_limits<DataT>::max();
    IndexT n_current_iter     = 0;
    DataT priorClusteringCost = 0;

    for (n_current_iter = 1; n_current_iter <= iter_params.max_iter; ++n_current_iter) {
      RAFT_LOG_DEBUG("KMeans.fit: Iteration-%d", n_current_iter);

      raft::matrix::fill(handle, centroid_sums.view(), DataT{0});
      raft::matrix::fill(handle, weight_per_cluster.view(), DataT{0});
      raft::linalg::map(handle,
                        raft::make_device_scalar_view(clustering_cost.data_handle()),
                        raft::const_op<DataT>{DataT{0}});

      auto centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
        centroidsRawData.data_handle(), n_clusters, n_features);

      data_batches.reset();
      weight_batches.reset();
      auto wt_it = weight_batches.begin();
      for (const auto& data_batch : data_batches) {
        IndexT cur_batch_size = static_cast<IndexT>(data_batch.size());
        const auto& wt_batch  = *wt_it;
        ++wt_it;

        auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
          data_batch.data(), cur_batch_size, n_features);
        auto batch_weights_view = prepare_batch_weights(wt_batch, cur_batch_size);

        auto minCAD_view = raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistance.data_handle(), cur_batch_size);
        auto l2_view =
          raft::make_device_vector_view<DataT, IndexT>(L2NormBatch.data_handle(), cur_batch_size);

        process_batch<DataT, IndexT>(handle,
                                     batch_data_view,
                                     batch_weights_view,
                                     centroids_const,
                                     metric,
                                     iter_params.batch_samples,
                                     iter_params.batch_centroids,
                                     minCAD_view,
                                     l2_view,
                                     L2NormBuf_OR_DistBuf,
                                     workspace,
                                     centroid_sums.view(),
                                     weight_per_cluster.view(),
                                     batch_sums.view(),
                                     batch_counts.view(),
                                     clustering_cost.view());
      }

      finalize_centroids<DataT, IndexT>(handle,
                                        raft::make_const_mdspan(centroid_sums.view()),
                                        raft::make_const_mdspan(weight_per_cluster.view()),
                                        centroids_const,
                                        new_centroids.view());

      DataT sqrdNormError =
        compute_centroid_shift<DataT, IndexT>(handle,
                                              raft::make_const_mdspan(centroids_const),
                                              raft::make_const_mdspan(new_centroids.view()));

      raft::copy(
        handle,
        raft::make_device_vector_view(centroidsRawData.data_handle(), new_centroids.size()),
        raft::make_device_vector_view(new_centroids.data_handle(), new_centroids.size()));

      bool done = false;

      DataT curClusteringCost = DataT{0};
      raft::copy(&curClusteringCost, clustering_cost.data_handle(), 1, stream);
      raft::resource::sync_stream(handle, stream);

      if (curClusteringCost == DataT{0}) {
        RAFT_LOG_WARN("Zero clustering cost detected: all points coincide with their centroids.");
      } else if (n_current_iter > 1) {
        DataT delta = curClusteringCost / priorClusteringCost;
        if (delta > 1 - iter_params.tol) done = true;
      }
      priorClusteringCost = curClusteringCost;

      if (sqrdNormError < iter_params.tol) done = true;

      if (done) {
        RAFT_LOG_DEBUG("Threshold triggered after %d iterations. Terminating early.",
                       n_current_iter);
        break;
      }
    }

    {
      auto centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
        centroidsRawData.data_handle(), n_clusters, n_features);

      iter_inertia = DataT{0};
      data_batches.reset();
      weight_batches.reset();
      auto wt_it = weight_batches.begin();
      for (const auto& data_batch : data_batches) {
        IndexT cur_batch_size = static_cast<IndexT>(data_batch.size());
        const auto& wt_batch  = *wt_it;
        ++wt_it;

        auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
          data_batch.data(), cur_batch_size, n_features);

        std::optional<raft::device_vector_view<const DataT, IndexT>> batch_sw = std::nullopt;
        if (weight_ptr != nullptr) { batch_sw = prepare_batch_weights(wt_batch, cur_batch_size); }

        DataT batch_cost = DataT{0};
        cuvs::cluster::kmeans::cluster_cost(handle,
                                            batch_data_view,
                                            centroids_const,
                                            raft::make_host_scalar_view(&batch_cost),
                                            batch_sw);

        iter_inertia += batch_cost;
      }
    }

    if (iter_inertia < inertia[0]) {
      inertia[0] = iter_inertia;
      n_iter[0]  = n_current_iter;
      raft::copy(
        handle,
        raft::make_device_vector_view(centroids.data_handle(), n_clusters * n_features),
        raft::make_device_vector_view(centroidsRawData.data_handle(), n_clusters * n_features));
    }
    RAFT_LOG_DEBUG("KMeans.fit after iteration-%d/%d: inertia - %f, n_iter - %d",
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
  auto XView = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroidsView =
    raft::make_device_matrix_view<DataT, IndexT>(centroids, pams.n_clusters, n_features);
  std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weightView = std::nullopt;
  if (sample_weight)
    sample_weightView =
      raft::make_device_vector_view<const DataT, IndexT>(sample_weight, n_samples);
  auto inertiaView = raft::make_host_scalar_view(&inertia);
  auto n_iterView  = raft::make_host_scalar_view(&n_iter);

  kmeans_fit(handle, pams, XView, sample_weightView, centroidsView, inertiaView, n_iterView);
}

template <typename DataT, typename IndexT>
void fit(raft::resources const& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const DataT, IndexT> X,
         std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  kmeans_fit(handle, params, X, sample_weight, centroids, inertia, n_iter);
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
  if (sample_weight.has_value())
    RAFT_EXPECTS(sample_weight.value().extent(0) == n_samples,
                 "invalid parameter (sample_weight!=n_samples)");
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
  if (sample_weight.has_value())
    raft::copy(handle, weight.view(), sample_weight.value());
  else
    raft::matrix::fill(handle, weight.view(), DataT(1));

  // check if weights sum up to n_samples
  if (normalize_weight) checkWeight(handle, weight.view(), workspace);

  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  // L2 norm of X: ||x||^2
  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle, X, L2NormX.view());
  }

  // computes minClusterAndDistance[0:n_samples) where  minClusterAndDistance[i]
  // is a <key, value> pair where
  //   'key' is index to a sample in 'centroids' (index of the nearest
  //   centroid) and 'value' is the distance between the sample 'X[i]' and the
  //   'centroid[key]'
  auto l2normx_view =
    raft::make_device_vector_view<const DataT, IndexT>(L2NormX.data_handle(), n_samples);
  cuvs::cluster::kmeans::detail::minClusterAndDistanceCompute<DataT, IndexT>(
    handle,
    X,
    centroids,
    minClusterAndDistance.view(),
    l2normx_view,
    L2NormBuf_OR_DistBuf,
    pams.metric,
    pams.batch_samples,
    pams.batch_centroids,
    workspace);

  // calculate cluster cost phi_x(C)
  rmm::device_scalar<DataT> clusterCostD(stream);
  raft::linalg::map(
    handle,
    minClusterAndDistance.view(),
    [=] __device__(const raft::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
      raft::KeyValuePair<IndexT, DataT> res;
      res.value = kvp.value * wt;
      res.key   = kvp.key;
      return res;
    },
    raft::make_const_mdspan(minClusterAndDistance.view()),
    raft::make_const_mdspan(weight.view()));

  cuvs::cluster::kmeans::detail::computeClusterCost(
    handle,
    minClusterAndDistance.view(),
    workspace,
    raft::make_device_scalar_view(clusterCostD.data()),
    raft::value_op{},
    raft::add_op{});

  raft::linalg::map(
    handle, labels, raft::key_op{}, raft::make_const_mdspan(minClusterAndDistance.view()));

  inertia[0] = clusterCostD.value(stream);
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
  auto XView = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroidsView =
    raft::make_device_matrix_view<const DataT, IndexT>(centroids, pams.n_clusters, n_features);
  std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weightView{std::nullopt};
  if (sample_weight)
    sample_weightView.emplace(
      raft::make_device_vector_view<const DataT, IndexT>(sample_weight, n_samples));
  auto labelsView  = raft::make_device_vector_view<IndexT, IndexT>(labels, n_samples);
  auto inertiaView = raft::make_host_scalar_view(&inertia);

  cuvs::cluster::kmeans::detail::kmeans_predict<DataT, IndexT>(handle,
                                                               pams,
                                                               XView,
                                                               sample_weightView,
                                                               centroidsView,
                                                               labelsView,
                                                               normalize_weight,
                                                               inertiaView);
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
  auto dataBatchSize = getDataBatchSize(pams.batch_samples, n_samples);

  // tile over the input data and calculate distance matrix [n_samples x
  // n_clusters]
  for (IndexT dIdx = 0; dIdx < (IndexT)n_samples; dIdx += dataBatchSize) {
    // # of samples for the current batch
    auto ns = std::min(static_cast<IndexT>(dataBatchSize), static_cast<IndexT>(n_samples - dIdx));

    // datasetView [ns x n_features] - view representing the current batch of
    // input dataset
    auto datasetView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + n_features * dIdx, ns, n_features);

    // pairwiseDistanceView [ns x n_clusters]
    auto pairwiseDistanceView = raft::make_device_matrix_view<DataT, IndexT>(
      X_new.data_handle() + n_clusters * dIdx, ns, n_clusters);

    // calculate pairwise distance between cluster centroids and current batch
    // of input dataset
    pairwise_distance_kmeans<DataT, IndexT>(
      handle, datasetView, centroids, pairwiseDistanceView, metric);
  }
}
}  // namespace cuvs::cluster::kmeans::detail
