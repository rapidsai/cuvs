/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../kmeans.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cstdint>
#include <random>

namespace cuvs::cluster::kmeans::mg::detail {

#define CUVS_LOG_KMEANS(handle, fmt, ...)                    \
  do {                                                       \
    bool isRoot = true;                                      \
    if (raft::resource::comms_initialized(handle)) {         \
      const auto& comm  = raft::resource::get_comms(handle); \
      const int my_rank = comm.get_rank();                   \
      isRoot            = my_rank == 0;                      \
    }                                                        \
    if (isRoot) { RAFT_LOG_DEBUG(fmt, ##__VA_ARGS__); }      \
  } while (0)

template <typename index_t, typename DataT>
struct key_value_index_op {
  __host__ __device__ __forceinline__ auto operator()(
    const raft::KeyValuePair<index_t, DataT>& a) const -> index_t
  {
    return a.key;
  }
};

#define KMEANS_COMM_ROOT 0

static cuvs::cluster::kmeans::params default_params;

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename index_t>
void init_random(const raft::resources& handle,
                 const cuvs::cluster::kmeans::params& params,
                 raft::device_matrix_view<const DataT, index_t> X,
                 raft::device_matrix_view<DataT, index_t> centroids)
{
  const auto& comm     = raft::resource::get_comms(handle);
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);
  auto n_clusters      = params.n_clusters;

  const int my_rank = comm.get_rank();
  const int n_ranks = comm.get_size();

  std::vector<int> n_centroids_sampled_by_rank(n_ranks, 0);
  std::vector<size_t> n_centroids_elements_to_receive_from_rank(n_ranks, 0);

  const int nranks_reqd = std::min(n_ranks, n_clusters);
  ASSERT(KMEANS_COMM_ROOT < nranks_reqd, "KMEANS_COMM_ROOT must be in [0,  %d)\n", nranks_reqd);

  for (int rank = 0; rank < nranks_reqd; ++rank) {
    int n_centroids_sampled_in_rank = n_clusters / nranks_reqd;
    if (rank == KMEANS_COMM_ROOT) {
      n_centroids_sampled_in_rank += n_clusters - n_centroids_sampled_in_rank * nranks_reqd;
    }
    n_centroids_sampled_by_rank[rank]               = n_centroids_sampled_in_rank;
    n_centroids_elements_to_receive_from_rank[rank] = n_centroids_sampled_in_rank * n_features;
  }

  auto n_centroids_sampled_in_rank = n_centroids_sampled_by_rank[my_rank];
  ASSERT((index_t)n_centroids_sampled_in_rank <= (index_t)n_local_samples,
         "# random samples requested from rank-%d is larger than the available "
         "samples at the rank (requested is %lu, available is %lu)",
         my_rank,
         (size_t)n_centroids_sampled_in_rank,
         (size_t)n_local_samples);

  auto centroids_sampled_in_rank =
    raft::make_device_matrix<DataT, index_t>(handle, n_centroids_sampled_in_rank, n_features);

  cuvs::cluster::kmeans::shuffle_and_gather(handle,
                                            X,
                                            centroids_sampled_in_rank.view(),
                                            n_centroids_sampled_in_rank,
                                            params.rng_state.seed);

  std::vector<size_t> displs(n_ranks);
  thrust::exclusive_scan(thrust::host,
                         n_centroids_elements_to_receive_from_rank.begin(),
                         n_centroids_elements_to_receive_from_rank.end(),
                         displs.begin());

  // gather centroids from all ranks
  comm.allgatherv<DataT>(centroids_sampled_in_rank.data_handle(),           // sendbuff
                         centroids.data_handle(),                           // recvbuff
                         n_centroids_elements_to_receive_from_rank.data(),  // recvcount
                         displs.data(),
                         stream);
}

/*
 * @brief Selects 'n_clusters' samples from X using scalable kmeans++ algorithm
 * Scalable kmeans++ pseudocode
 * 1: C = sample a point uniformly at random from X
 * 2: psi = phi_X (C)
 * 3: for O( log(psi) ) times do
 * 4:   C' = sample each point x in X independently with probability
 *           p_x = l * ( d^2(x, C) / phi_X (C) )
 * 5:   C = C U C'
 * 6: end for
 * 7: For x in C, set w_x to be the number of points in X closer to x than any
 *    other point in C
 * 8: Recluster the weighted points in C into k clusters
 */
template <typename DataT, typename index_t>
void init_k_means_plus_plus(const raft::resources& handle,
                            const cuvs::cluster::kmeans::params& params,
                            raft::device_matrix_view<const DataT, index_t> X,
                            raft::device_matrix_view<DataT, index_t> centroidsRawData,
                            rmm::device_uvector<char>& workspace)
{
  const auto& comm    = raft::resource::get_comms(handle);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  const int my_rank   = comm.get_rank();
  const int n_rank    = comm.get_size();

  auto n_samples  = X.extent(0);
  auto n_features = X.extent(1);
  auto n_clusters = params.n_clusters;
  auto metric     = params.metric;

  raft::random::RngState rng(params.rng_state.seed, raft::random::GeneratorType::GenPhilox);

  // <<<< Step-1 >>> : C <- sample a point uniformly at random from X
  //    1.1 - Select a rank r' at random from the available n_rank ranks with a
  //          probability of 1/n_rank [Note - with same seed all rank selects
  //          the same r' which avoids a call to comm]
  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  //    1.3 - Communicate the initial centroid chosen by rank-r' to all other
  //          ranks
  // Choose rp on rank 0 and broadcast to all ranks to guarantee agreement
  int rp = 0;
  if (my_rank == KMEANS_COMM_ROOT) {
    std::mt19937 gen(params.rng_state.seed);
    std::uniform_int_distribution<> dis(0, n_rank - 1);
    rp = dis(gen);
  }
  {
    rmm::device_scalar<int> rp_d(stream);
    raft::copy(rp_d.data(), &rp, 1, stream);
    comm.bcast<int>(rp_d.data(), 1, /*root=*/KMEANS_COMM_ROOT, stream);
    raft::copy(&rp, rp_d.data(), 1, stream);
    raft::resource::sync_stream(handle);
  }

  // buffer to flag the sample that is chosen as initial centroids
  std::vector<std::uint8_t> h_is_sample_centroid(n_samples);
  std::fill(h_is_sample_centroid.begin(), h_is_sample_centroid.end(), 0);

  auto initial_centroid = raft::make_device_matrix<DataT, index_t>(handle, 1, n_features);
  CUVS_LOG_KMEANS(
    handle, "@Rank-%d : KMeans|| : initial centroid is sampled at rank-%d\n", my_rank, rp);

  //    1.2 - Rank r' samples a point uniformly at random from the local dataset
  //          X which will be used as the initial centroid for kmeans++
  if (my_rank == rp) {
    std::mt19937 gen(params.rng_state.seed);
    std::uniform_int_distribution<> dis(0, n_samples - 1);

    int c_idx           = dis(gen);
    auto centroids_view = raft::make_device_matrix_view<const DataT, index_t>(
      X.data_handle() + c_idx * n_features, 1, n_features);

    raft::copy(
      initial_centroid.data_handle(), centroids_view.data_handle(), centroids_view.size(), stream);

    h_is_sample_centroid[c_idx] = 1;
  }

  // 1.3 - Communicate the initial centroid chosen by rank-r' to all other ranks
  comm.bcast<DataT>(initial_centroid.data_handle(), initial_centroid.size(), rp, stream);

  // device buffer to flag the sample that is chosen as initial centroid
  auto is_sample_centroid = raft::make_device_vector<std::uint8_t, index_t>(handle, n_samples);

  raft::copy(is_sample_centroid.data_handle(),
             h_is_sample_centroid.data(),
             is_sample_centroid.size(),
             stream);

  rmm::device_uvector<DataT> centroids_buf(0, stream);

  // reset buffer to store the chosen centroid
  centroids_buf.resize(initial_centroid.size(), stream);
  raft::copy(
    centroids_buf.begin(), initial_centroid.data_handle(), initial_centroid.size(), stream);

  auto potential_centroids = raft::make_device_matrix_view<DataT, index_t>(
    centroids_buf.data(), initial_centroid.extent(0), initial_centroid.extent(1));
  // <<< End of Step-1 >>>

  rmm::device_uvector<DataT> l2_norm_buf_or_dist_buf(0, stream);

  // L2 norm of X: ||x||^2
  auto l2_norm_x = raft::make_device_vector<DataT, index_t>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      l2_norm_x.data_handle(), X.data_handle(), X.extent(1), X.extent(0), stream);
  }

  auto min_cluster_distance = raft::make_device_vector<DataT, index_t>(handle, n_samples);
  auto uniform_rands        = raft::make_device_vector<DataT, index_t>(handle, n_samples);

  // <<< Step-2 >>>: psi <- phi_X (C)
  auto cluster_cost = raft::make_device_scalar<DataT>(handle, 0);

  cuvs::cluster::kmeans::min_cluster_distance(handle,
                                              X,
                                              potential_centroids,
                                              min_cluster_distance.view(),
                                              l2_norm_x.view(),
                                              l2_norm_buf_or_dist_buf,
                                              params.metric,
                                              params.batch_samples,
                                              params.batch_centroids,
                                              workspace);

  // compute partial cluster cost from the samples in rank
  cuvs::cluster::kmeans::cluster_cost(
    handle,
    min_cluster_distance.view(),
    workspace,
    cluster_cost.view(),
    cuda::proclaim_return_type<DataT>(
      [] __device__(const DataT& a, const DataT& b) -> DataT { return a + b; }));

  // compute total cluster cost by accumulating the partial cost from all the
  // ranks
  comm.allreduce(
    cluster_cost.data_handle(), cluster_cost.data_handle(), 1, raft::comms::op_t::SUM, stream);

  DataT psi = 0;
  raft::copy(&psi, cluster_cost.data_handle(), 1, stream);

  // <<< End of Step-2 >>>

  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");

  // Scalable kmeans++ paper claims 8 rounds is sufficient
  int niter = std::min(8, static_cast<int>(ceil(log(psi))));
  CUVS_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans|| :phi - %f, max # of iterations for kmeans++ loop - "
                  "%d\n",
                  my_rank,
                  psi,
                  niter);

  // <<<< Step-3 >>> : for O( log(psi) ) times do
  for (int iter = 0; iter < niter; ++iter) {
    CUVS_LOG_KMEANS(handle,
                    "@Rank-%d:KMeans|| - Iteration %d: # potential centroids sampled - "
                    "%d\n",
                    my_rank,
                    iter,
                    potential_centroids.extent(0));

    cuvs::cluster::kmeans::min_cluster_distance(handle,
                                                X,
                                                potential_centroids,
                                                min_cluster_distance.view(),
                                                l2_norm_x.view(),
                                                l2_norm_buf_or_dist_buf,
                                                params.metric,
                                                params.batch_samples,
                                                params.batch_centroids,
                                                workspace);

    cuvs::cluster::kmeans::cluster_cost(
      handle,
      min_cluster_distance.view(),
      workspace,
      cluster_cost.view(),
      cuda::proclaim_return_type<DataT>(
        [] __device__(const DataT& a, const DataT& b) -> DataT { return a + b; }));
    comm.allreduce(
      cluster_cost.data_handle(), cluster_cost.data_handle(), 1, raft::comms::op_t::SUM, stream);
    raft::copy(&psi, cluster_cost.data_handle(), 1, stream);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    // <<<< Step-4 >>> : Sample each point x in X independently and identify new
    // potential_centroids
    raft::random::uniform(handle,
                          rng,
                          uniform_rands.data_handle(),
                          uniform_rands.extent(0),
                          static_cast<DataT>(0),
                          static_cast<DataT>(1));
    cuvs::cluster::kmeans::sampling_op<DataT, index_t> select_op(psi,
                                                                 params.oversampling_factor,
                                                                 n_clusters,
                                                                 uniform_rands.data_handle(),
                                                                 is_sample_centroid.data_handle());

    rmm::device_uvector<DataT> in_rank_cp(0, stream);
    cuvs::cluster::kmeans::sample_centroids(handle,
                                            X,
                                            min_cluster_distance.view(),
                                            is_sample_centroid.view(),
                                            select_op,
                                            in_rank_cp,
                                            workspace);
    /// <<<< End of Step-4 >>>>

    int* n_pts_sampled_by_rank;
    RAFT_CUDA_TRY(cudaMallocHost(&n_pts_sampled_by_rank, n_rank * sizeof(int)));

    /// <<<< Step-5 >>> : C = C U C'
    // append the data in Cp from all ranks to the buffer holding the
    // potential_centroids
    // RAFT_CUDA_TRY(cudaMemsetAsync(n_pts_sampled_by_rank, 0, n_rank * sizeof(int), stream));
    std::fill(n_pts_sampled_by_rank, n_pts_sampled_by_rank + n_rank, 0);
    n_pts_sampled_by_rank[my_rank] = in_rank_cp.size() / n_features;
    comm.allgather(&(n_pts_sampled_by_rank[my_rank]), n_pts_sampled_by_rank, 1, stream);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    auto n_pts_sampled =
      thrust::reduce(thrust::host, n_pts_sampled_by_rank, n_pts_sampled_by_rank + n_rank, 0);

    // gather centroids from all ranks
    std::vector<size_t> sizes(n_rank);
    thrust::transform(thrust::host,
                      n_pts_sampled_by_rank,
                      n_pts_sampled_by_rank + n_rank,
                      sizes.begin(),
                      [&](int val) -> size_t { return val * n_features; });

    RAFT_CUDA_TRY_NO_THROW(cudaFreeHost(n_pts_sampled_by_rank));

    std::vector<size_t> displs(n_rank);
    thrust::exclusive_scan(thrust::host, sizes.begin(), sizes.end(), displs.begin());

    centroids_buf.resize(centroids_buf.size() + n_pts_sampled * n_features, stream);
    comm.allgatherv<DataT>(in_rank_cp.data(),
                           centroids_buf.end() - n_pts_sampled * n_features,
                           sizes.data(),
                           displs.data(),
                           stream);

    auto tot_centroids  = potential_centroids.extent(0) + n_pts_sampled;
    potential_centroids = raft::make_device_matrix_view<DataT, index_t>(
      centroids_buf.data(), tot_centroids, n_features);
    /// <<<< End of Step-5 >>>
  }  /// <<<< Step-6 >>>

  CUVS_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans||: # potential centroids sampled - %d\n",
                  my_rank,
                  potential_centroids.extent(0));

  if (static_cast<index_t>(potential_centroids.extent(0)) > static_cast<index_t>(n_clusters)) {
    // <<< Step-7 >>>: For x in C, set w_x to be the number of pts closest to X
    // temporary buffer to store the sample count per cluster, destructor
    // releases the resource

    auto weight = raft::make_device_vector<DataT, index_t>(handle, potential_centroids.extent(0));

    cuvs::cluster::kmeans::count_samples_in_cluster(
      handle, params, X, l2_norm_x.view(), potential_centroids, workspace, weight.view());

    // merge the local histogram from all ranks
    comm.allreduce<DataT>(weight.data_handle(),  // sendbuff
                          weight.data_handle(),  // recvbuff
                          weight.size(),         // count
                          raft::comms::op_t::SUM,
                          stream);

    // <<< end of Step-7 >>>

    // Step-8: Recluster the weighted points in C into k clusters
    // Note - reclustering step is duplicated across all ranks and with the same
    // seed they should generate the same potential_centroids
    auto const_centroids =
      raft::make_device_matrix_view<const DataT, index_t>(potential_centroids.data_handle(),
                                                          potential_centroids.extent(0),
                                                          potential_centroids.extent(1));
    cuvs::cluster::kmeans::init_plus_plus(
      handle, params, const_centroids, centroidsRawData, workspace);

    auto inertia = raft::make_host_scalar<DataT>(0);
    auto n_iter  = raft::make_host_scalar<index_t>(0);
    auto weight_view =
      raft::make_device_vector_view<const DataT, index_t>(weight.data_handle(), weight.extent(0));
    cuvs::cluster::kmeans::params params_copy = params;
    params_copy.rng_state                     = default_params.rng_state;

    cuvs::cluster::kmeans::fit_main<DataT, index_t>(handle,
                                                    params_copy,
                                                    const_centroids,
                                                    weight_view,
                                                    centroidsRawData,
                                                    inertia.view(),
                                                    n_iter.view(),
                                                    workspace);

  } else if (static_cast<index_t>(potential_centroids.extent(0)) <
             static_cast<index_t>(n_clusters)) {
    // supplement with random
    auto n_random_clusters = n_clusters - potential_centroids.extent(0);
    CUVS_LOG_KMEANS(handle,
                    "[Warning!] KMeans||: found fewer than %d centroids during "
                    "initialization (found %d centroids, remaining %d centroids will be "
                    "chosen randomly from input samples)\n",
                    n_clusters,
                    potential_centroids.extent(0),
                    n_random_clusters);

    // generate `n_random_clusters` centroids
    cuvs::cluster::kmeans::params rand_params = params;
    rand_params.rng_state                     = default_params.rng_state;
    rand_params.init                          = cuvs::cluster::kmeans::params::InitMethod::Random;
    rand_params.n_clusters                    = n_random_clusters;
    init_random(handle, rand_params, X, centroidsRawData);

    // copy centroids generated during kmeans|| iteration to the buffer
    raft::copy(centroidsRawData.data_handle() + n_random_clusters * n_features,
               potential_centroids.data_handle(),
               potential_centroids.size(),
               stream);

  } else {
    // found the required n_clusters
    raft::copy(centroidsRawData.data_handle(),
               potential_centroids.data_handle(),
               potential_centroids.size(),
               stream);
  }
}

template <typename DataT, typename index_t>
void check_weights(const raft::resources& handle,
                   rmm::device_uvector<char>& workspace,
                   raft::device_vector_view<DataT, index_t> weight)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  rmm::device_scalar<DataT> wt_aggr(stream);

  const auto& comm = raft::resource::get_comms(handle);

  auto n_samples            = weight.extent(0);
  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    nullptr, temp_storage_bytes, weight.data_handle(), wt_aggr.data(), n_samples, stream));

  workspace.resize(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
    workspace.data(), temp_storage_bytes, weight.data_handle(), wt_aggr.data(), n_samples, stream));

  comm.allreduce<DataT>(wt_aggr.data(),  // sendbuff
                        wt_aggr.data(),  // recvbuff
                        1,               // count
                        raft::comms::op_t::SUM,
                        stream);
  DataT wt_sum = wt_aggr.value(stream);
  raft::resource::sync_stream(handle, stream);

  if (wt_sum != n_samples) {
    CUVS_LOG_KMEANS(handle,
                    "[Warning!] KMeans: normalizing the user provided sample weights to "
                    "sum up to %d samples",
                    n_samples);

    DataT scale = n_samples / wt_sum;
    raft::linalg::unaryOp(weight.data_handle(),
                          weight.data_handle(),
                          weight.size(),
                          cuda::proclaim_return_type<DataT>(
                            [=] __device__(const DataT& wt) -> DataT { return wt * scale; }),
                          stream);
  }
}

template <typename DataT, typename index_t>
void fit(const raft::resources& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const DataT, index_t> X,
         std::optional<raft::device_vector_view<const DataT, index_t>> sample_weight,
         raft::device_matrix_view<DataT, index_t> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<index_t> n_iter,
         rmm::device_uvector<char>& workspace)
{
  const auto& comm    = raft::resource::get_comms(handle);
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_samples      = X.extent(0);
  auto n_features     = X.extent(1);
  auto n_clusters     = params.n_clusters;
  auto metric         = params.metric;

  auto weight = raft::make_device_vector<DataT, index_t>(handle, n_samples);
  if (sample_weight) {
    raft::copy(weight.data_handle(), sample_weight->data_handle(), n_samples, stream);
  } else {
    thrust::fill(raft::resource::get_thrust_policy(handle),
                 weight.data_handle(),
                 weight.data_handle() + weight.size(),
                 1);
  }

  // check if weights sum up to n_samples
  check_weights(handle, workspace, weight.view());

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
    // initializing with random samples from input dataset
    CUVS_LOG_KMEANS(handle,
                    "KMeans.fit: initialize cluster centers by randomly choosing from the "
                    "input data.\n");
    init_random<DataT, index_t>(handle, params, X, centroids);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
    // default method to initialize is kmeans++
    CUVS_LOG_KMEANS(handle, "KMeans.fit: initialize cluster centers using k-means++ algorithm.\n");
    init_k_means_plus_plus<DataT, index_t>(handle, params, X, centroids, workspace);
  } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
    CUVS_LOG_KMEANS(handle,
                    "KMeans.fit: initialize cluster centers from the ndarray array input "
                    "passed to init argument.\n");

  } else {
    THROW("unknown initialization method to select initial centers");
  }

  // stores (key, value) pair corresponding to each sample where
  //   - key is the index of nearest cluster
  //   - value is the distance to the nearest cluster
  auto min_cluster_and_distance =
    raft::make_device_vector<raft::KeyValuePair<index_t, DataT>, index_t>(handle, n_samples);

  // temporary buffer to store L2 norm of centroids or distance matrix,
  // destructor releases the resource
  rmm::device_uvector<DataT> l2_norm_buf_or_dist_buf(0, stream);

  // temporary buffer to store intermediate centroids, destructor releases the
  // resource
  auto new_centroids = raft::make_device_matrix<DataT, index_t>(handle, n_clusters, n_features);

  // temporary buffer to store the weights per cluster, destructor releases
  // the resource
  auto wt_in_cluster = raft::make_device_vector<DataT, index_t>(handle, n_clusters);

  // L2 norm of X: ||x||^2
  auto l2_norm_x = raft::make_device_vector<DataT, index_t>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      l2_norm_x.data_handle(), X.data_handle(), X.extent(1), X.extent(0), stream);
  }

  DataT prior_clustering_cost = 0;
  for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
    CUVS_LOG_KMEANS(handle,
                    "KMeans.fit: Iteration-%d: fitting the model using the initialize "
                    "cluster centers\n",
                    n_iter[0]);

    auto const_centroids = raft::make_device_matrix_view<const DataT, index_t>(
      centroids.data_handle(), centroids.extent(0), centroids.extent(1));
    // computes min_cluster_and_distance[0:n_samples) where
    // min_cluster_and_distance[i] is a <key, value> pair where
    //   'key' is index to an sample in 'centroids' (index of the nearest
    //   centroid) and 'value' is the distance between the sample 'X[i]' and the
    //   'centroid[key]'
    cuvs::cluster::kmeans::min_cluster_and_distance(handle,
                                                    X,
                                                    const_centroids,
                                                    min_cluster_and_distance.view(),
                                                    l2_norm_x.view(),
                                                    l2_norm_buf_or_dist_buf,
                                                    params.metric,
                                                    params.batch_samples,
                                                    params.batch_centroids,
                                                    workspace);

    // Using TransformInputIteratorT to dereference an array of
    // cub::KeyValuePair and converting them to just return the Key to be used
    // in reduce_rows_by_key prims
    key_value_index_op<index_t, DataT> conversion_op;
    thrust::transform_iterator<key_value_index_op<index_t, DataT>,
                               raft::KeyValuePair<index_t, DataT>*>
      itr(min_cluster_and_distance.data_handle(), conversion_op);

    workspace.resize(n_samples, stream);

    // Calculates weighted sum of all the samples assigned to cluster-i and
    // store the result in new_centroids[i]
    raft::linalg::reduce_rows_by_key(const_cast<DataT*>(X.data_handle()),
                                     X.extent(1),
                                     itr,
                                     weight.data_handle(),
                                     workspace.data(),
                                     X.extent(0),
                                     X.extent(1),
                                     static_cast<index_t>(n_clusters),
                                     new_centroids.data_handle(),
                                     stream);

    // Reduce weights by key to compute weight in each cluster
    raft::linalg::reduce_cols_by_key(weight.data_handle(),
                                     itr,
                                     wt_in_cluster.data_handle(),
                                     static_cast<index_t>(1),
                                     static_cast<index_t>(weight.extent(0)),
                                     static_cast<index_t>(n_clusters),
                                     stream);

    // merge the local histogram from all ranks
    comm.allreduce<DataT>(wt_in_cluster.data_handle(),  // sendbuff
                          wt_in_cluster.data_handle(),  // recvbuff
                          wt_in_cluster.size(),         // count
                          raft::comms::op_t::SUM,
                          stream);

    // reduces new_centroids from all ranks
    comm.allreduce<DataT>(new_centroids.data_handle(),  // sendbuff
                          new_centroids.data_handle(),  // recvbuff
                          new_centroids.size(),         // count
                          raft::comms::op_t::SUM,
                          stream);

    // Computes new_centroids[i] = new_centroids[i]/wt_in_cluster[i] where
    //   new_centroids[n_clusters x n_features] - 2D array, new_centroids[i] has
    //   sum of all the samples assigned to cluster-i
    //   wt_in_cluster[n_clusters] - 1D array, wt_in_cluster[i] contains # of
    //   samples in cluster-i.
    // Note - when wt_in_cluster[i] is 0, newCentroid[i] is reset to 0

    raft::linalg::matrix_vector_op<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_const_mdspan(new_centroids.view()),
      raft::make_const_mdspan(wt_in_cluster.view()),
      new_centroids.view(),
      cuda::proclaim_return_type<DataT>([=] __device__(DataT mat, DataT vec) -> DataT {
        if (vec == 0) {
          return DataT(0);
        } else {
          return mat / vec;
        }
      }));

    // copy the centroids[i] to new_centroids[i] when wt_in_cluster[i] is 0
    cub::ArgIndexInputIterator<DataT*> itr_wt(wt_in_cluster.data_handle());
    raft::matrix::gather_if(
      centroids.data_handle(),
      centroids.extent(1),
      centroids.extent(0),
      itr_wt,
      itr_wt,
      wt_in_cluster.extent(0),
      new_centroids.data_handle(),
      cuda::proclaim_return_type<bool>(
        [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) -> bool {  // predicate
          // copy when the # of samples in the cluster is 0
          if (map.value == 0) {
            return true;
          } else {
            return false;
          }
        }),
      cuda::proclaim_return_type<ptrdiff_t>(
        [=] __device__(raft::KeyValuePair<ptrdiff_t, DataT> map) {  // map
          return map.key;
        }),
      stream);

    // compute the squared norm between the new_centroids and the original
    // centroids, destructor releases the resource
    auto sqrd_norm = raft::make_device_scalar<DataT>(handle, 1);
    raft::linalg::mapThenSumReduce(
      sqrd_norm.data_handle(),
      new_centroids.size(),
      cuda::proclaim_return_type<DataT>([=] __device__(const DataT a, const DataT b) -> DataT {
        DataT diff = a - b;
        return diff * diff;
      }),
      stream,
      centroids.data_handle(),
      new_centroids.data_handle());

    DataT sqrd_norm_error = 0;
    raft::copy(&sqrd_norm_error, sqrd_norm.data_handle(), sqrd_norm.size(), stream);

    raft::copy(centroids.data_handle(), new_centroids.data_handle(), new_centroids.size(), stream);

    bool done = false;
    if (params.inertia_check) {
      rmm::device_scalar<raft::KeyValuePair<index_t, DataT>> cluster_cost_d(stream);

      // calculate cluster cost phi_x(C)
      cuvs::cluster::kmeans::cluster_cost(
        handle,
        min_cluster_and_distance.view(),
        workspace,
        raft::make_device_scalar_view(cluster_cost_d.data()),
        cuda::proclaim_return_type<raft::KeyValuePair<index_t, DataT>>(
          [] __device__(
            const raft::KeyValuePair<index_t, DataT>& a,
            const raft::KeyValuePair<index_t, DataT>& b) -> raft::KeyValuePair<index_t, DataT> {
            raft::KeyValuePair<index_t, DataT> res;
            res.key   = 0;
            res.value = a.value + b.value;
            return res;
          }));

      // Cluster cost phi_x(C) from all ranks
      comm.allreduce(&(cluster_cost_d.data()->value),
                     &(cluster_cost_d.data()->value),
                     1,
                     raft::comms::op_t::SUM,
                     stream);

      DataT cur_clustering_cost = 0;
      raft::copy(&cur_clustering_cost, &(cluster_cost_d.data()->value), 1, stream);

      ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
             "An error occurred in the distributed operation. This can result "
             "from a failed rank");
      ASSERT(cur_clustering_cost != (DataT)0.0,
             "Too few points and centroids being found is getting 0 cost from "
             "centers\n");

      if (n_iter[0] > 0) {
        DataT delta = cur_clustering_cost / prior_clustering_cost;
        if (delta > 1 - params.tol) done = true;
      }
      prior_clustering_cost = cur_clustering_cost;
    }

    raft::resource::sync_stream(handle, stream);
    if (sqrd_norm_error < params.tol) done = true;

    if (done) {
      CUVS_LOG_KMEANS(
        handle, "Threshold triggered after %d iterations. Terminating early.\n", n_iter[0]);
      break;
    }
  }
}

};  // namespace cuvs::cluster::kmeans::mg::detail
