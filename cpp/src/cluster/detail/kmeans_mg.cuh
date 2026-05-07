/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../kmeans.cuh"
#include "kmeans_common.cuh"

#include "../../neighbors/detail/ann_utils.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/functional>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

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

#define KMEANS_COMM_ROOT 0

static cuvs::cluster::kmeans::params default_params;

// =========================================================================
// Reduce operation abstraction
// =========================================================================

/**
 * @brief Reduce functor that wraps raft::comms for MNMG (multi-process) use.
 *
 * Any backend that provides allreduce/bcast/group_start/group_end with
 * the same signatures can be used to instantiate fit_impl (e.g. raw NCCL
 * for SNMG).
 */
struct raft_comms_reduce_op {
  const raft::resources& handle_;

  explicit raft_comms_reduce_op(const raft::resources& h) : handle_(h) {}

  template <typename T>
  void allreduce(T* sendbuf, T* recvbuf, size_t count, cudaStream_t stream) const
  {
    const auto& comm = raft::resource::get_comms(handle_);
    comm.allreduce(sendbuf, recvbuf, count, raft::comms::op_t::SUM, stream);
  }

  template <typename T>
  void bcast(T* buf, size_t count, int root, cudaStream_t stream) const
  {
    const auto& comm = raft::resource::get_comms(handle_);
    comm.bcast(buf, count, root, stream);
  }

  void group_start() const {}
  void group_end() const {}

  int get_rank() const
  {
    const auto& comm = raft::resource::get_comms(handle_);
    return comm.get_rank();
  }

  int get_num_ranks() const
  {
    const auto& comm = raft::resource::get_comms(handle_);
    return comm.get_size();
  }

  void sync(cudaStream_t stream) const
  {
    const auto& comm = raft::resource::get_comms(handle_);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. "
           "This can result from a failed rank");
  }
};

// =========================================================================
// Initialization helpers
// =========================================================================

// Selects 'n_clusters' samples randomly from X
template <typename DataT, typename IndexT>
void initRandom(const raft::resources& handle,
                const cuvs::cluster::kmeans::params& params,
                raft::device_matrix_view<const DataT, IndexT> X,
                raft::device_matrix_view<DataT, IndexT> centroids)
{
  const auto& comm     = raft::resource::get_comms(handle);
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto n_local_samples = X.extent(0);
  auto n_features      = X.extent(1);
  auto n_clusters      = params.n_clusters;

  const int my_rank = comm.get_rank();
  const int n_ranks = comm.get_size();

  std::vector<int> nCentroidsSampledByRank(n_ranks, 0);
  std::vector<size_t> nCentroidsElementsToReceiveFromRank(n_ranks, 0);

  const int nranks_reqd = std::min(n_ranks, n_clusters);
  ASSERT(KMEANS_COMM_ROOT < nranks_reqd, "KMEANS_COMM_ROOT must be in [0,  %d)\n", nranks_reqd);

  for (int rank = 0; rank < nranks_reqd; ++rank) {
    int nCentroidsSampledInRank = n_clusters / nranks_reqd;
    if (rank == KMEANS_COMM_ROOT) {
      nCentroidsSampledInRank += n_clusters - nCentroidsSampledInRank * nranks_reqd;
    }
    nCentroidsSampledByRank[rank]             = nCentroidsSampledInRank;
    nCentroidsElementsToReceiveFromRank[rank] = nCentroidsSampledInRank * n_features;
  }

  auto nCentroidsSampledInRank = nCentroidsSampledByRank[my_rank];
  ASSERT((IndexT)nCentroidsSampledInRank <= (IndexT)n_local_samples,
         "# random samples requested from rank-%d is larger than the available "
         "samples at the rank (requested is %lu, available is %lu)",
         my_rank,
         (size_t)nCentroidsSampledInRank,
         (size_t)n_local_samples);

  auto centroidsSampledInRank =
    raft::make_device_matrix<DataT, IndexT>(handle, nCentroidsSampledInRank, n_features);

  cuvs::cluster::kmeans::shuffle_and_gather(
    handle, X, centroidsSampledInRank.view(), nCentroidsSampledInRank, params.rng_state.seed);

  std::vector<size_t> displs(n_ranks);
  std::exclusive_scan(nCentroidsElementsToReceiveFromRank.begin(),
                      nCentroidsElementsToReceiveFromRank.end(),
                      displs.begin(),
                      size_t(0));

  comm.allgatherv<DataT>(centroidsSampledInRank.data_handle(),
                         centroids.data_handle(),
                         nCentroidsElementsToReceiveFromRank.data(),
                         displs.data(),
                         stream);
}

template <typename DataT, typename IndexT>
void initKMeansPlusPlus(const raft::resources& handle,
                        const cuvs::cluster::kmeans::params& params,
                        raft::device_matrix_view<const DataT, IndexT> X,
                        raft::device_matrix_view<DataT, IndexT> centroidsRawData,
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

  int rp = 0;
  if (my_rank == KMEANS_COMM_ROOT) {
    std::mt19937 gen(params.rng_state.seed);
    std::uniform_int_distribution<> dis(0, n_rank - 1);
    rp = dis(gen);
  }
  {
    rmm::device_scalar<int> rp_d(stream);
    raft::copy(
      handle, raft::make_device_scalar_view(rp_d.data()), raft::make_host_scalar_view(&rp));
    comm.bcast<int>(rp_d.data(), 1, KMEANS_COMM_ROOT, stream);
    raft::copy(
      handle, raft::make_host_scalar_view(&rp), raft::make_device_scalar_view(rp_d.data()));
    raft::resource::sync_stream(handle);
  }

  std::vector<std::uint8_t> h_isSampleCentroid(n_samples);
  std::fill(h_isSampleCentroid.begin(), h_isSampleCentroid.end(), 0);

  auto initialCentroid = raft::make_device_matrix<DataT, IndexT>(handle, 1, n_features);
  CUVS_LOG_KMEANS(
    handle, "@Rank-%d : KMeans|| : initial centroid is sampled at rank-%d\n", my_rank, rp);

  if (my_rank == rp) {
    std::mt19937 gen(params.rng_state.seed);
    std::uniform_int_distribution<> dis(0, n_samples - 1);

    int cIdx           = dis(gen);
    auto centroidsView = raft::make_device_matrix_view<const DataT, IndexT>(
      X.data_handle() + cIdx * n_features, 1, n_features);

    raft::copy(handle,
               raft::make_device_vector_view(initialCentroid.data_handle(), centroidsView.size()),
               raft::make_device_vector_view(centroidsView.data_handle(), centroidsView.size()));

    h_isSampleCentroid[cIdx] = 1;
  }

  comm.bcast<DataT>(initialCentroid.data_handle(), initialCentroid.size(), rp, stream);

  auto isSampleCentroid = raft::make_device_vector<std::uint8_t, IndexT>(handle, n_samples);

  raft::copy(handle,
             raft::make_device_vector_view(isSampleCentroid.data_handle(), isSampleCentroid.size()),
             raft::make_host_vector_view(h_isSampleCentroid.data(), isSampleCentroid.size()));

  rmm::device_uvector<DataT> centroidsBuf(0, stream);

  centroidsBuf.resize(initialCentroid.size(), stream);
  raft::copy(handle,
             raft::make_device_vector_view(centroidsBuf.begin(), initialCentroid.size()),
             raft::make_device_vector_view(initialCentroid.data_handle(), initialCentroid.size()));

  auto potentialCentroids = raft::make_device_matrix_view<DataT, IndexT>(
    centroidsBuf.data(), initialCentroid.extent(0), initialCentroid.extent(1));

  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

  auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  if (metric == cuvs::distance::DistanceType::L2Expanded ||
      metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_device_matrix_view<const DataT, IndexT, raft::row_major>(
        X.data_handle(), n_samples, n_features),
      L2NormX.view());
  }

  auto minClusterDistance = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
  auto uniformRands       = raft::make_device_vector<DataT, IndexT>(handle, n_samples);

  auto clusterCost = raft::make_device_scalar<DataT>(handle, 0);

  cuvs::cluster::kmeans::min_cluster_distance(handle,
                                              X,
                                              potentialCentroids,
                                              minClusterDistance.view(),
                                              L2NormX.view(),
                                              L2NormBuf_OR_DistBuf,
                                              params.metric,
                                              params.batch_samples,
                                              params.batch_centroids,
                                              workspace);

  cuvs::cluster::kmeans::cluster_cost(
    handle,
    minClusterDistance.view(),
    workspace,
    clusterCost.view(),
    cuda::proclaim_return_type<DataT>(
      [] __device__(const DataT& a, const DataT& b) { return a + b; }));

  comm.allreduce(
    clusterCost.data_handle(), clusterCost.data_handle(), 1, raft::comms::op_t::SUM, stream);

  DataT psi = 0;
  raft::copy(handle, raft::make_host_scalar_view(&psi), clusterCost.view());

  ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
         "An error occurred in the distributed operation. This can result from "
         "a failed rank");

  int niter = std::min(8, (int)ceil(log(psi)));
  CUVS_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans|| :phi - %f, max # of iterations for kmeans++ loop - "
                  "%d\n",
                  my_rank,
                  psi,
                  niter);

  for (int iter = 0; iter < niter; ++iter) {
    CUVS_LOG_KMEANS(handle,
                    "@Rank-%d:KMeans|| - Iteration %d: # potential centroids sampled - "
                    "%d\n",
                    my_rank,
                    iter,
                    potentialCentroids.extent(0));

    cuvs::cluster::kmeans::min_cluster_distance(handle,
                                                X,
                                                potentialCentroids,
                                                minClusterDistance.view(),
                                                L2NormX.view(),
                                                L2NormBuf_OR_DistBuf,
                                                params.metric,
                                                params.batch_samples,
                                                params.batch_centroids,
                                                workspace);

    cuvs::cluster::kmeans::cluster_cost(
      handle,
      minClusterDistance.view(),
      workspace,
      clusterCost.view(),
      cuda::proclaim_return_type<DataT>(
        [] __device__(const DataT& a, const DataT& b) { return a + b; }));
    comm.allreduce(
      clusterCost.data_handle(), clusterCost.data_handle(), 1, raft::comms::op_t::SUM, stream);
    raft::copy(handle, raft::make_host_scalar_view(&psi), clusterCost.view());
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    raft::random::uniform(
      handle, rng, uniformRands.data_handle(), uniformRands.extent(0), (DataT)0, (DataT)1);
    cuvs::cluster::kmeans::SamplingOp<DataT, IndexT> select_op(psi,
                                                               params.oversampling_factor,
                                                               n_clusters,
                                                               uniformRands.data_handle(),
                                                               isSampleCentroid.data_handle());

    rmm::device_uvector<DataT> inRankCp(0, stream);
    cuvs::cluster::kmeans::sample_centroids(handle,
                                            X,
                                            minClusterDistance.view(),
                                            isSampleCentroid.view(),
                                            select_op,
                                            inRankCp,
                                            workspace);

    int* nPtsSampledByRank;
    RAFT_CUDA_TRY(cudaMallocHost(&nPtsSampledByRank, n_rank * sizeof(int)));

    std::fill(nPtsSampledByRank, nPtsSampledByRank + n_rank, 0);
    nPtsSampledByRank[my_rank] = inRankCp.size() / n_features;
    comm.allgather(&(nPtsSampledByRank[my_rank]), nPtsSampledByRank, 1, stream);
    ASSERT(comm.sync_stream(stream) == raft::comms::status_t::SUCCESS,
           "An error occurred in the distributed operation. This can result "
           "from a failed rank");

    auto nPtsSampled = std::reduce(nPtsSampledByRank, nPtsSampledByRank + n_rank, 0);

    std::vector<size_t> sizes(n_rank);
    std::transform(
      nPtsSampledByRank, nPtsSampledByRank + n_rank, sizes.begin(), [n_features](int val) {
        return static_cast<size_t>(val) * n_features;
      });

    RAFT_CUDA_TRY_NO_THROW(cudaFreeHost(nPtsSampledByRank));

    std::vector<size_t> displs(n_rank);
    std::exclusive_scan(sizes.begin(), sizes.end(), displs.begin(), size_t(0));

    centroidsBuf.resize(centroidsBuf.size() + nPtsSampled * n_features, stream);
    comm.allgatherv<DataT>(inRankCp.data(),
                           centroidsBuf.end() - nPtsSampled * n_features,
                           sizes.data(),
                           displs.data(),
                           stream);

    auto tot_centroids = potentialCentroids.extent(0) + nPtsSampled;
    potentialCentroids =
      raft::make_device_matrix_view<DataT, IndexT>(centroidsBuf.data(), tot_centroids, n_features);
  }

  CUVS_LOG_KMEANS(handle,
                  "@Rank-%d:KMeans||: # potential centroids sampled - %d\n",
                  my_rank,
                  potentialCentroids.extent(0));

  if ((IndexT)potentialCentroids.extent(0) > (IndexT)n_clusters) {
    auto weight = raft::make_device_vector<DataT, IndexT>(handle, potentialCentroids.extent(0));

    cuvs::cluster::kmeans::count_samples_in_cluster(
      handle, params, X, L2NormX.view(), potentialCentroids, workspace, weight.view());

    comm.allreduce<DataT>(
      weight.data_handle(), weight.data_handle(), weight.size(), raft::comms::op_t::SUM, stream);

    auto const_centroids = raft::make_device_matrix_view<const DataT, IndexT>(
      potentialCentroids.data_handle(), potentialCentroids.extent(0), potentialCentroids.extent(1));
    cuvs::cluster::kmeans::init_plus_plus(
      handle, params, const_centroids, centroidsRawData, workspace);

    auto inertia = raft::make_host_scalar<DataT>(0);
    auto n_iter  = raft::make_host_scalar<IndexT>(0);
    auto weight_view =
      raft::make_device_vector_view<const DataT, IndexT>(weight.data_handle(), weight.extent(0));
    cuvs::cluster::kmeans::params params_copy = params;
    params_copy.rng_state                     = default_params.rng_state;

    cuvs::cluster::kmeans::fit_main<DataT, IndexT>(handle,
                                                   params_copy,
                                                   const_centroids,
                                                   weight_view,
                                                   centroidsRawData,
                                                   inertia.view(),
                                                   n_iter.view(),
                                                   workspace);

  } else if ((IndexT)potentialCentroids.extent(0) < (IndexT)n_clusters) {
    auto n_random_clusters = n_clusters - potentialCentroids.extent(0);
    CUVS_LOG_KMEANS(handle,
                    "[Warning!] KMeans||: found fewer than %d centroids during "
                    "initialization (found %d centroids, remaining %d centroids will be "
                    "chosen randomly from input samples)\n",
                    n_clusters,
                    potentialCentroids.extent(0),
                    n_random_clusters);

    cuvs::cluster::kmeans::params rand_params = params;
    rand_params.rng_state                     = default_params.rng_state;
    rand_params.init                          = cuvs::cluster::kmeans::params::InitMethod::Random;
    rand_params.n_clusters                    = n_random_clusters;
    initRandom(handle, rand_params, X, centroidsRawData);

    raft::copy(
      handle,
      raft::make_device_vector_view(centroidsRawData.data_handle() + n_random_clusters * n_features,
                                    potentialCentroids.size()),
      raft::make_device_vector_view(potentialCentroids.data_handle(), potentialCentroids.size()));

  } else {
    raft::copy(
      handle,
      raft::make_device_vector_view(centroidsRawData.data_handle(), potentialCentroids.size()),
      raft::make_device_vector_view(potentialCentroids.data_handle(), potentialCentroids.size()));
  }
}

// =========================================================================
// Unified fit_impl — core Lloyd iterations
// =========================================================================

/**
 * @brief Unified multi-GPU kmeans fit implementation.
 *
 * Templated on the mdspan type of X and sample_weight so that a single
 * implementation handles both host data (streaming in batches) and device
 * data (single batch). The template parameter ReduceOp abstracts the
 * communication backend:
 *   - raft_comms_reduce_op  for MNMG (raft::comms, one-process-per-GPU)
 *   - A raw-NCCL reduce op  for SNMG (OpenMP, one-thread-per-GPU)
 *
 * @tparam XMatrixView   raft::host_matrix_view or raft::device_matrix_view
 * @tparam SWVectorView  raft::host_vector_view or raft::device_vector_view
 * @tparam DataT         float or double
 * @tparam IndexT        int or int64_t
 * @tparam ReduceOp      communication functor type
 */
template <typename XMatrixView,
          typename SWVectorView,
          typename DataT,
          typename IndexT,
          typename ReduceOp>
void fit_impl(const raft::resources& handle,
              const cuvs::cluster::kmeans::params& params,
              const ReduceOp& reduce_op,
              XMatrixView X,
              std::optional<SWVectorView> sample_weight,
              raft::device_matrix_view<DataT, IndexT> centroids,
              raft::host_scalar_view<DataT> inertia,
              raft::host_scalar_view<IndexT> n_iter)
{
  constexpr bool streaming = raft::is_host_mdspan_v<XMatrixView>;

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int rank            = reduce_op.get_rank();

  auto n_local    = static_cast<IndexT>(X.extent(0));
  auto n_features = static_cast<IndexT>(X.extent(1));
  auto n_clusters = static_cast<IndexT>(params.n_clusters);
  auto metric     = params.metric;

  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IndexT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(centroids.extent(1) == n_features, "centroids.extent(1) must equal n_features");

  IndexT streaming_batch_size = static_cast<IndexT>(params.streaming_batch_size);
  if constexpr (!streaming) {
    streaming_batch_size = n_local;
  } else {
    if (streaming_batch_size <= 0 || streaming_batch_size > n_local) {
      streaming_batch_size = std::max(n_local, IndexT{1});
    }
  }

  bool has_data = (n_local > 0);

  // --- Weight normalization across ranks ---
  DataT weight_scale = DataT{1};
  if (sample_weight.has_value()) {
    if constexpr (streaming) {
      DataT local_wt_sum = DataT{0};
      const DataT* sw    = sample_weight->data_handle();
      for (IndexT i = 0; i < n_local; ++i)
        local_wt_sum += sw[i];

      auto d_local_n = raft::make_device_scalar<DataT>(handle, static_cast<DataT>(n_local));
      auto d_wt      = raft::make_device_scalar<DataT>(handle, local_wt_sum);
      reduce_op.allreduce(d_local_n.data_handle(), d_local_n.data_handle(), 1, stream);
      reduce_op.allreduce(d_wt.data_handle(), d_wt.data_handle(), 1, stream);
      raft::resource::sync_stream(handle, stream);

      DataT global_n{}, global_wt{};
      raft::copy(&global_n, d_local_n.data_handle(), 1, stream);
      raft::copy(&global_wt, d_wt.data_handle(), 1, stream);
      raft::resource::sync_stream(handle, stream);
      if (global_wt != global_n) { weight_scale = global_n / global_wt; }
    } else {
      auto wt = raft::make_device_vector<DataT, IndexT>(handle, n_local);
      raft::copy(handle, wt.view(), sample_weight.value());
      rmm::device_uvector<char> ws(0, stream);
      checkWeights(handle, reduce_op, ws, wt.view());
    }
  }

  // --- Allocate work buffers ---
  IndexT alloc_batch_size = has_data ? streaming_batch_size : IndexT{1};
  IndexT weights_size     = streaming ? alloc_batch_size : std::max(n_local, IndexT{1});

  auto rank_centroids     = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto new_centroids      = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto centroid_sums      = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto weight_per_cluster = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);
  auto clustering_cost    = raft::make_device_scalar<DataT>(handle, DataT{0});
  auto batch_weights      = raft::make_device_vector<DataT, IndexT>(handle, weights_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, alloc_batch_size);
  auto L2NormBatch = raft::make_device_vector<DataT, IndexT>(handle, alloc_batch_size);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_uvector<char> workspace(0, stream);
  rmm::device_uvector<char> batch_workspace(0, stream);

  auto d_done = raft::make_device_scalar<int64_t>(handle, 0);

  // --- Initialization ---
  if constexpr (streaming) {
    if (params.init != cuvs::cluster::kmeans::params::InitMethod::Array) {
      IndexT init_sample_size = std::min(static_cast<IndexT>(3 * n_clusters), n_local);
      auto init_sample =
        raft::make_device_matrix<DataT, IndexT>(handle, init_sample_size, n_features);
      raft::random::RngState rng(params.rng_state.seed);
      raft::matrix::sample_rows(handle, rng, X, init_sample.view());

      if (rank == 0) {
        auto init_const = raft::make_device_matrix_view<const DataT, IndexT>(
          init_sample.data_handle(), init_sample_size, n_features);
        if (params.oversampling_factor == 0)
          cuvs::cluster::kmeans::detail::kmeansPlusPlus<DataT, IndexT>(
            handle, params, init_const, rank_centroids.view(), workspace);
        else
          cuvs::cluster::kmeans::detail::initScalableKMeansPlusPlus<DataT, IndexT>(
            handle, params, init_const, rank_centroids.view(), workspace);
      }
      raft::resource::sync_stream(handle, stream);
      reduce_op.bcast(
        rank_centroids.data_handle(), static_cast<size_t>(n_clusters * n_features), 0, stream);
      raft::resource::sync_stream(handle, stream);
    } else {
      raft::copy(
        rank_centroids.data_handle(), centroids.data_handle(), n_clusters * n_features, stream);
    }
  } else {
    if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
      initRandom<DataT, IndexT>(handle, params, X, rank_centroids.view());
    } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
      initKMeansPlusPlus<DataT, IndexT>(handle, params, X, rank_centroids.view(), workspace);
    } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
      raft::copy(
        rank_centroids.data_handle(), centroids.data_handle(), n_clusters * n_features, stream);
    } else {
      THROW("unknown initialization method to select initial centers");
    }
  }

  // --- Prepare device-side weights ---
  if constexpr (!streaming) {
    if (sample_weight.has_value()) {
      raft::copy(handle, batch_weights.view(), sample_weight.value());
      if (weight_scale != DataT{1}) {
        raft::linalg::map(handle,
                          batch_weights.view(),
                          raft::mul_const_op<DataT>{weight_scale},
                          raft::make_const_mdspan(batch_weights.view()));
      }
    } else {
      raft::matrix::fill(handle, batch_weights.view(), DataT{1});
    }
  } else {
    if (!sample_weight.has_value()) { raft::matrix::fill(handle, batch_weights.view(), DataT{1}); }
  }

  // --- Pre-compute norms for device data ---
  bool need_norms = metric == cuvs::distance::DistanceType::L2Expanded ||
                    metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                    metric == cuvs::distance::DistanceType::CosineExpanded;
  if constexpr (!streaming) {
    if (has_data && need_norms) {
      auto norm_view =
        raft::make_device_vector_view<DataT, IndexT>(L2NormBatch.data_handle(), n_local);
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
          handle, X, norm_view, raft::sqrt_op{});
      } else {
        raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(handle, X, norm_view);
      }
    }
  }

  // --- Batch iterator for streaming path ---
  std::optional<cuvs::spatial::knn::detail::utils::batch_load_iterator<DataT>> data_batches_opt;
  if constexpr (streaming) {
    if (has_data) {
      data_batches_opt.emplace(X.data_handle(),
                               static_cast<size_t>(n_local),
                               static_cast<size_t>(n_features),
                               static_cast<size_t>(streaming_batch_size),
                               stream,
                               rmm::mr::get_current_device_resource_ref(),
                               true);
    }
  }

  DataT prior_cluster_cost = DataT{0};
  IndexT local_n_iter      = 0;

  // =====================================================================
  // Lloyd iterations
  // =====================================================================
  for (local_n_iter = 1; local_n_iter <= params.max_iter; ++local_n_iter) {
    RAFT_LOG_DEBUG("MG KMeans: iteration %d on rank %d", local_n_iter, rank);

    raft::matrix::fill(handle, centroid_sums.view(), DataT{0});
    raft::matrix::fill(handle, weight_per_cluster.view(), DataT{0});
    raft::linalg::map(handle,
                      raft::make_device_scalar_view(clustering_cost.data_handle()),
                      raft::const_op<DataT>{DataT{0}});

    auto rank_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      rank_centroids.data_handle(), n_clusters, n_features);

    // --- Phase 1: Local accumulation ---
    if (has_data) {
      if constexpr (streaming) {
        auto& data_batches = *data_batches_opt;
        data_batches.reset();
        data_batches.prefetch_next_batch();
        for (const auto& data_batch : data_batches) {
          IndexT cur_batch_size = static_cast<IndexT>(data_batch.size());

          auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
            data_batch.data(), cur_batch_size, n_features);

          if (sample_weight.has_value()) {
            raft::copy(batch_weights.data_handle(),
                       sample_weight->data_handle() + data_batch.offset(),
                       cur_batch_size,
                       stream);
            if (weight_scale != DataT{1}) {
              auto bw = raft::make_device_vector_view<DataT, IndexT>(batch_weights.data_handle(),
                                                                     cur_batch_size);
              raft::linalg::map(
                handle, bw, raft::mul_const_op<DataT>{weight_scale}, raft::make_const_mdspan(bw));
            }
          }

          auto batch_weights_view = raft::make_device_vector_view<const DataT, IndexT>(
            batch_weights.data_handle(), cur_batch_size);

          auto L2NormBatch_view =
            raft::make_device_vector_view<DataT, IndexT>(L2NormBatch.data_handle(), cur_batch_size);

          if (need_norms) {
            auto bv = raft::make_device_matrix_view<const DataT, IndexT>(
              data_batch.data(), cur_batch_size, n_features);
            if (metric == cuvs::distance::DistanceType::CosineExpanded) {
              raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                handle, bv, L2NormBatch_view, raft::sqrt_op{});
            } else {
              raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                handle, bv, L2NormBatch_view);
            }
          }

          auto L2NormBatch_const = raft::make_device_vector_view<const DataT, IndexT>(
            L2NormBatch.data_handle(), cur_batch_size);
          auto minCAD_view =
            raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
              minClusterAndDistance.data_handle(), cur_batch_size);

          cuvs::cluster::kmeans::detail::process_batch<DataT, IndexT>(
            handle,
            batch_data_view,
            batch_weights_view,
            rank_centroids_const,
            metric,
            params.batch_samples,
            params.batch_centroids,
            minCAD_view,
            L2NormBatch_const,
            L2NormBuf_OR_DistBuf,
            workspace,
            centroid_sums.view(),
            weight_per_cluster.view(),
            raft::make_device_scalar_view(clustering_cost.data_handle()),
            batch_workspace);

          data_batches.prefetch_next_batch();
        }
      } else {
        auto batch_weights_view =
          raft::make_device_vector_view<const DataT, IndexT>(batch_weights.data_handle(), n_local);
        auto L2NormBatch_const =
          raft::make_device_vector_view<const DataT, IndexT>(L2NormBatch.data_handle(), n_local);
        auto minCAD_view = raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
          minClusterAndDistance.data_handle(), n_local);

        cuvs::cluster::kmeans::detail::process_batch<DataT, IndexT>(
          handle,
          X,
          batch_weights_view,
          rank_centroids_const,
          metric,
          params.batch_samples,
          params.batch_centroids,
          minCAD_view,
          L2NormBatch_const,
          L2NormBuf_OR_DistBuf,
          workspace,
          centroid_sums.view(),
          weight_per_cluster.view(),
          raft::make_device_scalar_view(clustering_cost.data_handle()),
          batch_workspace);
      }
    }

    // --- Phase 2: Allreduce partial sums ---
    reduce_op.group_start();
    reduce_op.allreduce(centroid_sums.data_handle(),
                        centroid_sums.data_handle(),
                        static_cast<size_t>(n_clusters * n_features),
                        stream);
    reduce_op.allreduce(weight_per_cluster.data_handle(),
                        weight_per_cluster.data_handle(),
                        static_cast<size_t>(n_clusters),
                        stream);
    reduce_op.allreduce(
      clustering_cost.data_handle(), clustering_cost.data_handle(), size_t{1}, stream);
    reduce_op.group_end();
    raft::resource::sync_stream(handle, stream);

    // --- Phase 3: Finalize centroids ---
    cuvs::cluster::kmeans::detail::finalize_centroids<DataT, IndexT>(
      handle,
      raft::make_const_mdspan(centroid_sums.view()),
      raft::make_const_mdspan(weight_per_cluster.view()),
      rank_centroids_const,
      new_centroids.view());

    // --- Phase 4: Convergence check ---
    auto d_sqrdNormError = raft::make_device_scalar<DataT>(handle, DataT{0});
    cuvs::cluster::kmeans::detail::compute_centroid_shift<DataT, IndexT>(
      handle,
      raft::make_const_mdspan(rank_centroids.view()),
      raft::make_const_mdspan(new_centroids.view()),
      d_sqrdNormError.view());
    DataT sqrdNormError = DataT{0};
    raft::copy(&sqrdNormError, d_sqrdNormError.data_handle(), 1, stream);

    raft::copy(
      rank_centroids.data_handle(), new_centroids.data_handle(), n_clusters * n_features, stream);

    bool done = false;

    DataT curClusteringCost = DataT{0};
    raft::copy(&curClusteringCost, clustering_cost.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);

    if (curClusteringCost == DataT{0}) {
      RAFT_LOG_WARN("Zero clustering cost detected: all points coincide with their centroids.");
    } else if (local_n_iter > 1) {
      DataT delta = curClusteringCost / prior_cluster_cost;
      if (delta > 1 - params.tol) { done = true; }
    }
    prior_cluster_cost = curClusteringCost;

    if (sqrdNormError < params.tol) { done = true; }

    int64_t done_val = done ? 1 : 0;
    raft::copy(d_done.data_handle(), &done_val, 1, stream);
    raft::resource::sync_stream(handle, stream);
    reduce_op.allreduce(d_done.data_handle(), d_done.data_handle(), size_t{1}, stream);
    raft::resource::sync_stream(handle, stream);
    raft::copy(&done_val, d_done.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    done = (done_val > 0);

    if (done) {
      RAFT_LOG_DEBUG(
        "MG KMeans: threshold triggered after %d iterations on rank %d", local_n_iter, rank);
      break;
    }
  }
  if (local_n_iter > static_cast<IndexT>(params.max_iter)) {
    local_n_iter = static_cast<IndexT>(params.max_iter);
  }

  // --- Final inertia computation ---
  raft::linalg::map(handle,
                    raft::make_device_scalar_view(clustering_cost.data_handle()),
                    raft::const_op<DataT>{DataT{0}});

  if (has_data) {
    auto rank_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      rank_centroids.data_handle(), n_clusters, n_features);

    if constexpr (streaming) {
      auto& data_batches = *data_batches_opt;
      data_batches.reset();
      data_batches.prefetch_next_batch();
      for (const auto& data_batch : data_batches) {
        IndexT cur_batch_size = static_cast<IndexT>(data_batch.size());

        auto batch_data_view = raft::make_device_matrix_view<const DataT, IndexT>(
          data_batch.data(), cur_batch_size, n_features);

        std::optional<raft::device_vector_view<const DataT, IndexT>> batch_sw = std::nullopt;
        if (sample_weight.has_value()) {
          raft::copy(batch_weights.data_handle(),
                     sample_weight->data_handle() + data_batch.offset(),
                     cur_batch_size,
                     stream);
          if (weight_scale != DataT{1}) {
            auto bw = raft::make_device_vector_view<DataT, IndexT>(batch_weights.data_handle(),
                                                                   cur_batch_size);
            raft::linalg::map(
              handle, bw, raft::mul_const_op<DataT>{weight_scale}, raft::make_const_mdspan(bw));
          }
          batch_sw = raft::make_device_vector_view<const DataT, IndexT>(batch_weights.data_handle(),
                                                                        cur_batch_size);
        }

        DataT batch_cost_h = DataT{0};
        cuvs::cluster::kmeans::cluster_cost(handle,
                                            batch_data_view,
                                            rank_centroids_const,
                                            raft::make_host_scalar_view(&batch_cost_h),
                                            batch_sw);

        auto d_batch_cost = raft::make_device_scalar<DataT>(handle, batch_cost_h);
        raft::linalg::add(clustering_cost.data_handle(),
                          clustering_cost.data_handle(),
                          d_batch_cost.data_handle(),
                          1,
                          stream);

        data_batches.prefetch_next_batch();
      }
    } else {
      std::optional<raft::device_vector_view<const DataT, IndexT>> dev_sw = std::nullopt;
      if (sample_weight.has_value()) {
        dev_sw =
          raft::make_device_vector_view<const DataT, IndexT>(batch_weights.data_handle(), n_local);
      }
      DataT batch_cost_h = DataT{0};
      cuvs::cluster::kmeans::cluster_cost(
        handle, X, rank_centroids_const, raft::make_host_scalar_view(&batch_cost_h), dev_sw);
      auto d_batch_cost = raft::make_device_scalar<DataT>(handle, batch_cost_h);
      raft::linalg::add(clustering_cost.data_handle(),
                        clustering_cost.data_handle(),
                        d_batch_cost.data_handle(),
                        1,
                        stream);
    }
  }

  reduce_op.allreduce(
    clustering_cost.data_handle(), clustering_cost.data_handle(), size_t{1}, stream);
  raft::resource::sync_stream(handle, stream);
  raft::copy(&inertia[0], clustering_cost.data_handle(), 1, stream);
  raft::resource::sync_stream(handle, stream);

  raft::copy(
    centroids.data_handle(), rank_centroids.data_handle(), n_clusters * n_features, stream);
  raft::resource::sync_stream(handle, stream);
  n_iter[0] = local_n_iter;
}

// =========================================================================
// Weight checking (generalized for any reduce op)
// =========================================================================
template <typename DataT, typename IndexT, typename ReduceOp>
void checkWeights(const raft::resources& handle,
                  const ReduceOp& reduce_op,
                  rmm::device_uvector<char>& workspace,
                  raft::device_vector_view<DataT, IndexT> weight)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  rmm::device_scalar<DataT> wt_aggr(stream);

  const auto& comm = raft::resource::get_comms(handle);

  auto n_samples = weight.extent(0);
  raft::linalg::mapThenSumReduce(
    wt_aggr.data(), n_samples, raft::identity_op{}, stream, weight.data_handle());

  comm.allreduce<DataT>(wt_aggr.data(),  // sendbuff
                        wt_aggr.data(),  // recvbuff
                        1,               // count
                        raft::comms::op_t::SUM,
                        stream);
  DataT wt_sum = wt_aggr.value(stream);
  raft::resource::sync_stream(handle, stream);
  RAFT_EXPECTS(wt_sum > DataT{0}, "invalid parameter (sum of sample weights must be positive)");

  const auto target = static_cast<DataT>(n_samples);
  const DataT tol   = target * std::numeric_limits<DataT>::epsilon();
  if (std::abs(wt_sum - target) > tol) {
    CUVS_LOG_KMEANS(handle,
                    "[Warning!] KMeans: normalizing the user provided sample weights to "
                    "sum up to %d samples",
                    n_samples);

    raft::linalg::map(handle,
                      weight,
                      raft::compose_op(raft::mul_const_op<DataT>{static_cast<DataT>(n_samples)},
                                       raft::div_const_op<DataT>{wt_sum}),
                      raft::make_const_mdspan(weight));
  }
}

// =========================================================================
// Host-partitioned MNMG fit implementation
// =========================================================================

template <typename DataT, typename IndexT>
using host_matrix_parts_t = std::vector<raft::host_matrix_view<const DataT, IndexT>>;

template <typename DataT, typename IndexT>
using host_weight_parts_t = std::vector<raft::host_vector_view<const DataT, IndexT>>;

template <typename DataT, typename IndexT>
void sample_host_partitions(const raft::resources& handle,
                            const host_matrix_parts_t<DataT, IndexT>& X_parts,
                            IndexT n_local,
                            raft::device_matrix_view<DataT, IndexT> sample,
                            uint64_t seed)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_rows         = static_cast<IndexT>(sample.extent(0));
  auto n_features     = static_cast<IndexT>(sample.extent(1));

  RAFT_EXPECTS(n_local > 0, "cannot sample centroids from empty local host partitions");

  std::vector<IndexT> offsets;
  offsets.reserve(X_parts.size() + 1);
  offsets.push_back(IndexT{0});
  for (auto const& X : X_parts) {
    offsets.push_back(offsets.back() + static_cast<IndexT>(X.extent(0)));
  }

  std::vector<DataT> h_sample(static_cast<size_t>(n_rows) * n_features);
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<IndexT> dist(IndexT{0}, n_local - 1);

  for (IndexT row = 0; row < n_rows; ++row) {
    IndexT global_row = dist(gen);
    auto upper        = std::upper_bound(offsets.begin(), offsets.end(), global_row);
    size_t part_idx   = static_cast<size_t>(std::distance(offsets.begin(), upper) - 1);
    IndexT local_row  = global_row - offsets[part_idx];

    auto const* src = X_parts[part_idx].data_handle() + local_row * n_features;
    auto* dst       = h_sample.data() + row * n_features;
    std::copy_n(src, n_features, dst);
  }

  raft::update_device(sample.data_handle(), h_sample.data(), h_sample.size(), stream);
}

template <typename DataT, typename IndexT, typename ReduceOp>
void fit_host_partitions_impl(
  const raft::resources& handle,
  const cuvs::cluster::kmeans::params& params,
  const ReduceOp& reduce_op,
  const host_matrix_parts_t<DataT, IndexT>& X_parts,
  const std::optional<host_weight_parts_t<DataT, IndexT>>& sample_weight_parts,
  raft::device_matrix_view<DataT, IndexT> centroids,
  raft::host_scalar_view<DataT> inertia,
  raft::host_scalar_view<IndexT> n_iter)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int rank            = reduce_op.get_rank();

  auto n_clusters = static_cast<IndexT>(params.n_clusters);
  auto n_features = static_cast<IndexT>(centroids.extent(1));
  auto metric     = params.metric;

  RAFT_EXPECTS(n_clusters > 0, "n_clusters must be positive");
  RAFT_EXPECTS(static_cast<IndexT>(centroids.extent(0)) == n_clusters,
               "centroids.extent(0) must equal n_clusters");
  RAFT_EXPECTS(n_features > 0, "centroids.extent(1) must be positive");

  IndexT n_local = 0;
  for (auto const& X : X_parts) {
    RAFT_EXPECTS(static_cast<IndexT>(X.extent(1)) == n_features,
                 "all host partitions must have the same feature count as centroids");
    n_local += static_cast<IndexT>(X.extent(0));
  }

  if (sample_weight_parts.has_value()) {
    RAFT_EXPECTS(sample_weight_parts->size() == X_parts.size(),
                 "sample_weight_parts must have one entry per host partition");
    for (size_t i = 0; i < X_parts.size(); ++i) {
      RAFT_EXPECTS(static_cast<IndexT>((*sample_weight_parts)[i].extent(0)) ==
                     static_cast<IndexT>(X_parts[i].extent(0)),
                   "each sample_weight partition must match its X partition rows");
    }
  }

  auto d_global_n = raft::make_device_scalar<IndexT>(handle, n_local);
  reduce_op.allreduce(d_global_n.data_handle(), d_global_n.data_handle(), size_t{1}, stream);
  raft::resource::sync_stream(handle, stream);
  IndexT global_n = 0;
  raft::copy(&global_n, d_global_n.data_handle(), 1, stream);
  raft::resource::sync_stream(handle, stream);
  RAFT_EXPECTS(global_n > 0, "at least one sample is required across all ranks");

  IndexT streaming_batch_size = static_cast<IndexT>(params.streaming_batch_size);
  if (streaming_batch_size <= 0 || streaming_batch_size > n_local) {
    streaming_batch_size = std::max(n_local, IndexT{1});
  }

  bool has_data = n_local > 0;

  DataT weight_scale = DataT{1};
  if (sample_weight_parts.has_value()) {
    DataT local_wt_sum = DataT{0};
    for (auto const& weights : *sample_weight_parts) {
      auto n_weights = static_cast<IndexT>(weights.extent(0));
      for (IndexT i = 0; i < n_weights; ++i) {
        local_wt_sum += weights.data_handle()[i];
      }
    }

    auto d_wt = raft::make_device_scalar<DataT>(handle, local_wt_sum);
    reduce_op.allreduce(d_wt.data_handle(), d_wt.data_handle(), size_t{1}, stream);
    raft::resource::sync_stream(handle, stream);
    DataT global_wt = DataT{0};
    raft::copy(&global_wt, d_wt.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    RAFT_EXPECTS(std::isfinite(global_wt) && global_wt > DataT{0},
                 "invalid parameter (sum of sample weights must be finite and positive)");

    DataT target = static_cast<DataT>(global_n);
    DataT tol    = target * std::numeric_limits<DataT>::epsilon();
    if (std::abs(global_wt - target) > tol) { weight_scale = target / global_wt; }
  }

  IndexT alloc_batch_size = has_data ? streaming_batch_size : IndexT{1};
  auto rank_centroids     = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto new_centroids      = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto centroid_sums      = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);
  auto weight_per_cluster = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);
  auto clustering_cost    = raft::make_device_scalar<DataT>(handle, DataT{0});
  auto batch_weights      = raft::make_device_vector<DataT, IndexT>(handle, alloc_batch_size);
  auto minClusterAndDistance =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, alloc_batch_size);
  auto L2NormBatch = raft::make_device_vector<DataT, IndexT>(handle, alloc_batch_size);
  rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);
  rmm::device_uvector<char> workspace(0, stream);
  rmm::device_uvector<char> batch_workspace(0, stream);
  auto d_done = raft::make_device_scalar<int64_t>(handle, 0);

  if (params.init == cuvs::cluster::kmeans::params::InitMethod::Array) {
    raft::copy(
      rank_centroids.data_handle(), centroids.data_handle(), n_clusters * n_features, stream);
  } else {
    if (rank == KMEANS_COMM_ROOT) {
      RAFT_EXPECTS(n_local > 0,
                   "rank 0 must own at least one sample for host-partitioned initialization");
      if (params.init == cuvs::cluster::kmeans::params::InitMethod::Random) {
        sample_host_partitions<DataT, IndexT>(
          handle, X_parts, n_local, rank_centroids.view(), params.rng_state.seed);
      } else if (params.init == cuvs::cluster::kmeans::params::InitMethod::KMeansPlusPlus) {
        IndexT init_sample_size = std::max(n_clusters, std::min(IndexT{3} * n_clusters, n_local));
        auto init_sample =
          raft::make_device_matrix<DataT, IndexT>(handle, init_sample_size, n_features);
        sample_host_partitions<DataT, IndexT>(
          handle, X_parts, n_local, init_sample.view(), params.rng_state.seed);

        auto init_const = raft::make_device_matrix_view<const DataT, IndexT>(
          init_sample.data_handle(), init_sample_size, n_features);
        if (params.oversampling_factor == 0) {
          cuvs::cluster::kmeans::detail::kmeansPlusPlus<DataT, IndexT>(
            handle, params, init_const, rank_centroids.view(), workspace);
        } else {
          cuvs::cluster::kmeans::detail::initScalableKMeansPlusPlus<DataT, IndexT>(
            handle, params, init_const, rank_centroids.view(), workspace);
        }
      } else {
        THROW("unknown initialization method to select initial centers");
      }
    }
    raft::resource::sync_stream(handle, stream);
    reduce_op.bcast(rank_centroids.data_handle(),
                    static_cast<size_t>(n_clusters * n_features),
                    KMEANS_COMM_ROOT,
                    stream);
    raft::resource::sync_stream(handle, stream);
  }

  if (!sample_weight_parts.has_value()) {
    raft::matrix::fill(handle, batch_weights.view(), DataT{1});
  }

  bool need_norms = metric == cuvs::distance::DistanceType::L2Expanded ||
                    metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                    metric == cuvs::distance::DistanceType::CosineExpanded;

  auto prepare_batch_weights = [&](size_t part_idx, size_t batch_offset, IndexT cur_batch_size) {
    if (sample_weight_parts.has_value()) {
      raft::copy(batch_weights.data_handle(),
                 (*sample_weight_parts)[part_idx].data_handle() + batch_offset,
                 cur_batch_size,
                 stream);
      if (weight_scale != DataT{1}) {
        auto bw =
          raft::make_device_vector_view<DataT, IndexT>(batch_weights.data_handle(), cur_batch_size);
        raft::linalg::map(
          handle, bw, raft::mul_const_op<DataT>{weight_scale}, raft::make_const_mdspan(bw));
      }
    }
    return raft::make_device_vector_view<const DataT, IndexT>(batch_weights.data_handle(),
                                                              cur_batch_size);
  };

  auto process_local_partitions =
    [&](raft::device_matrix_view<const DataT, IndexT> rank_centroids_const) {
      for (size_t part_idx = 0; part_idx < X_parts.size(); ++part_idx) {
        auto const& X_part = X_parts[part_idx];
        auto part_rows     = static_cast<IndexT>(X_part.extent(0));
        if (part_rows == 0) { continue; }

        cuvs::spatial::knn::detail::utils::batch_load_iterator<DataT> data_batches(
          X_part.data_handle(),
          static_cast<size_t>(part_rows),
          static_cast<size_t>(n_features),
          static_cast<size_t>(streaming_batch_size),
          stream,
          rmm::mr::get_current_device_resource_ref(),
          true);
        data_batches.prefetch_next_batch();

        for (auto const& data_batch : data_batches) {
          IndexT cur_batch_size = static_cast<IndexT>(data_batch.size());
          auto batch_data_view  = raft::make_device_matrix_view<const DataT, IndexT>(
            data_batch.data(), cur_batch_size, n_features);
          auto batch_weights_view =
            prepare_batch_weights(part_idx, data_batch.offset(), cur_batch_size);
          auto L2NormBatch_view =
            raft::make_device_vector_view<DataT, IndexT>(L2NormBatch.data_handle(), cur_batch_size);

          if (need_norms) {
            if (metric == cuvs::distance::DistanceType::CosineExpanded) {
              raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                handle, batch_data_view, L2NormBatch_view, raft::sqrt_op{});
            } else {
              raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
                handle, batch_data_view, L2NormBatch_view);
            }
          }

          auto minCAD_view =
            raft::make_device_vector_view<raft::KeyValuePair<IndexT, DataT>, IndexT>(
              minClusterAndDistance.data_handle(), cur_batch_size);
          auto L2NormBatch_const = raft::make_device_vector_view<const DataT, IndexT>(
            L2NormBatch.data_handle(), cur_batch_size);

          cuvs::cluster::kmeans::detail::process_batch<DataT, IndexT>(handle,
                                                                      batch_data_view,
                                                                      batch_weights_view,
                                                                      rank_centroids_const,
                                                                      metric,
                                                                      params.batch_samples,
                                                                      params.batch_centroids,
                                                                      minCAD_view,
                                                                      L2NormBatch_const,
                                                                      L2NormBuf_OR_DistBuf,
                                                                      workspace,
                                                                      centroid_sums.view(),
                                                                      weight_per_cluster.view(),
                                                                      clustering_cost.view(),
                                                                      batch_workspace);

          data_batches.prefetch_next_batch();
        }
      }
    };

  DataT prior_cluster_cost = DataT{0};
  IndexT local_n_iter      = 0;

  for (local_n_iter = 1; local_n_iter <= params.max_iter; ++local_n_iter) {
    RAFT_LOG_DEBUG("MG KMeans partitions: iteration %d on rank %d", local_n_iter, rank);

    raft::matrix::fill(handle, centroid_sums.view(), DataT{0});
    raft::matrix::fill(handle, weight_per_cluster.view(), DataT{0});
    raft::linalg::map(handle,
                      raft::make_device_scalar_view(clustering_cost.data_handle()),
                      raft::const_op<DataT>{DataT{0}});

    auto rank_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
      rank_centroids.data_handle(), n_clusters, n_features);
    if (has_data) { process_local_partitions(rank_centroids_const); }

    reduce_op.group_start();
    reduce_op.allreduce(centroid_sums.data_handle(),
                        centroid_sums.data_handle(),
                        static_cast<size_t>(n_clusters * n_features),
                        stream);
    reduce_op.allreduce(weight_per_cluster.data_handle(),
                        weight_per_cluster.data_handle(),
                        static_cast<size_t>(n_clusters),
                        stream);
    reduce_op.allreduce(
      clustering_cost.data_handle(), clustering_cost.data_handle(), size_t{1}, stream);
    reduce_op.group_end();
    raft::resource::sync_stream(handle, stream);

    cuvs::cluster::kmeans::detail::finalize_centroids<DataT, IndexT>(
      handle,
      raft::make_const_mdspan(centroid_sums.view()),
      raft::make_const_mdspan(weight_per_cluster.view()),
      rank_centroids_const,
      new_centroids.view());

    auto d_sqrdNormError = raft::make_device_scalar<DataT>(handle, DataT{0});
    cuvs::cluster::kmeans::detail::compute_centroid_shift<DataT, IndexT>(
      handle,
      raft::make_const_mdspan(rank_centroids.view()),
      raft::make_const_mdspan(new_centroids.view()),
      d_sqrdNormError.view());
    DataT sqrdNormError = DataT{0};
    raft::copy(&sqrdNormError, d_sqrdNormError.data_handle(), 1, stream);

    raft::copy(
      rank_centroids.data_handle(), new_centroids.data_handle(), n_clusters * n_features, stream);

    DataT curClusteringCost = DataT{0};
    raft::copy(&curClusteringCost, clustering_cost.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);

    bool done = false;
    if (curClusteringCost == DataT{0}) {
      RAFT_LOG_WARN("Zero clustering cost detected: all points coincide with their centroids.");
    } else if (local_n_iter > 1 && prior_cluster_cost > DataT{0}) {
      DataT delta = curClusteringCost / prior_cluster_cost;
      if (delta > DataT{1} - params.tol) { done = true; }
    }
    prior_cluster_cost = curClusteringCost;
    if (sqrdNormError < params.tol) { done = true; }

    int64_t done_val = done ? 1 : 0;
    raft::copy(d_done.data_handle(), &done_val, 1, stream);
    raft::resource::sync_stream(handle, stream);
    reduce_op.allreduce(d_done.data_handle(), d_done.data_handle(), size_t{1}, stream);
    raft::resource::sync_stream(handle, stream);
    raft::copy(&done_val, d_done.data_handle(), 1, stream);
    raft::resource::sync_stream(handle, stream);
    if (done_val > 0) { break; }
  }
  if (local_n_iter > static_cast<IndexT>(params.max_iter)) {
    local_n_iter = static_cast<IndexT>(params.max_iter);
  }

  raft::linalg::map(handle,
                    raft::make_device_scalar_view(clustering_cost.data_handle()),
                    raft::const_op<DataT>{DataT{0}});

  auto rank_centroids_const = raft::make_device_matrix_view<const DataT, IndexT>(
    rank_centroids.data_handle(), n_clusters, n_features);
  if (has_data) {
    for (size_t part_idx = 0; part_idx < X_parts.size(); ++part_idx) {
      auto const& X_part = X_parts[part_idx];
      auto part_rows     = static_cast<IndexT>(X_part.extent(0));
      if (part_rows == 0) { continue; }

      cuvs::spatial::knn::detail::utils::batch_load_iterator<DataT> data_batches(
        X_part.data_handle(),
        static_cast<size_t>(part_rows),
        static_cast<size_t>(n_features),
        static_cast<size_t>(streaming_batch_size),
        stream,
        rmm::mr::get_current_device_resource_ref(),
        true);
      data_batches.prefetch_next_batch();

      for (auto const& data_batch : data_batches) {
        IndexT cur_batch_size = static_cast<IndexT>(data_batch.size());
        auto batch_data_view  = raft::make_device_matrix_view<const DataT, IndexT>(
          data_batch.data(), cur_batch_size, n_features);

        std::optional<raft::device_vector_view<const DataT, IndexT>> batch_sw = std::nullopt;
        if (sample_weight_parts.has_value()) {
          batch_sw = prepare_batch_weights(part_idx, data_batch.offset(), cur_batch_size);
        }

        DataT batch_cost_h = DataT{0};
        cuvs::cluster::kmeans::cluster_cost(handle,
                                            batch_data_view,
                                            rank_centroids_const,
                                            raft::make_host_scalar_view(&batch_cost_h),
                                            batch_sw);
        auto d_batch_cost = raft::make_device_scalar<DataT>(handle, batch_cost_h);
        raft::linalg::add(clustering_cost.data_handle(),
                          clustering_cost.data_handle(),
                          d_batch_cost.data_handle(),
                          1,
                          stream);

        data_batches.prefetch_next_batch();
      }
    }
  }

  reduce_op.allreduce(
    clustering_cost.data_handle(), clustering_cost.data_handle(), size_t{1}, stream);
  raft::resource::sync_stream(handle, stream);
  raft::copy(&inertia[0], clustering_cost.data_handle(), 1, stream);
  raft::resource::sync_stream(handle, stream);

  raft::copy(
    centroids.data_handle(), rank_centroids.data_handle(), n_clusters * n_features, stream);
  raft::resource::sync_stream(handle, stream);
  n_iter[0] = local_n_iter;
}

// =========================================================================
// Public entry points
// =========================================================================

/**
 * @brief MNMG kmeans fit with device data (existing API).
 */
template <typename DataT, typename IndexT>
void fit(const raft::resources& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::device_matrix_view<const DataT, IndexT> X,
         std::optional<raft::device_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter,
         rmm::device_uvector<char>& /*workspace*/)
{
  raft_comms_reduce_op reduce_op(handle);
  fit_impl(handle, params, reduce_op, X, sample_weight, centroids, inertia, n_iter);
}

/**
 * @brief MNMG kmeans fit with host data (streaming).
 *
 * Each rank provides its local host-resident data partition. Data is streamed
 * to the GPU in batches controlled by params.streaming_batch_size.
 */
template <typename DataT, typename IndexT>
void fit(const raft::resources& handle,
         const cuvs::cluster::kmeans::params& params,
         raft::host_matrix_view<const DataT, IndexT> X,
         std::optional<raft::host_vector_view<const DataT, IndexT>> sample_weight,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  raft_comms_reduce_op reduce_op(handle);
  fit_impl(handle, params, reduce_op, X, sample_weight, centroids, inertia, n_iter);
}

/**
 * @brief MNMG kmeans fit with multiple local host data partitions.
 *
 * Each rank may provide zero or more local host-resident partitions. Every
 * partition is streamed to the rank's GPU in batches controlled by
 * params.streaming_batch_size, and local accumulations are reduced globally
 * once per Lloyd iteration.
 */
template <typename DataT, typename IndexT>
void fit(const raft::resources& handle,
         const cuvs::cluster::kmeans::params& params,
         const host_matrix_parts_t<DataT, IndexT>& X_parts,
         const std::optional<host_weight_parts_t<DataT, IndexT>>& sample_weight_parts,
         raft::device_matrix_view<DataT, IndexT> centroids,
         raft::host_scalar_view<DataT> inertia,
         raft::host_scalar_view<IndexT> n_iter)
{
  raft_comms_reduce_op reduce_op(handle);
  fit_host_partitions_impl(
    handle, params, reduce_op, X_parts, sample_weight_parts, centroids, inertia, n_iter);
}

}  // namespace cuvs::cluster::kmeans::mg::detail
