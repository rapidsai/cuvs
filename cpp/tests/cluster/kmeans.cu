/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

namespace cuvs {

template <typename T>
struct KmeansInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

// template <typename DataT, typename IndexT>
// void run_cluster_cost(const raft::resources& handle,
//                       raft::device_vector_view<DataT, IndexT> minClusterDistance,
//                       rmm::device_uvector<char>& workspace,
//                       raft::device_scalar_view<DataT> clusterCost)
//{
//   cuvs::cluster::kmeans::cluster_cost(
//     handle, minClusterDistance, workspace, clusterCost, raft::add_op{});
// }

template <typename T>
class KmeansTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansTest()
    : d_labels(0, raft::resource::get_cuda_stream(handle)),
      d_labels_ref(0, raft::resource::get_cuda_stream(handle)),
      d_centroids(0, raft::resource::get_cuda_stream(handle)),
      d_sample_weight(0, raft::resource::get_cuda_stream(handle))
  {
  }

  //  void apiTest()
  //  {
  //    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
  //
  //    auto stream                = raft::resource::get_cuda_stream(handle);
  //    int n_samples              = testparams.n_row;
  //    int n_features             = testparams.n_col;
  //    params.n_clusters          = testparams.n_clusters;
  //    params.tol                 = testparams.tol;
  //    params.n_init              = 1;
  //    params.rng_state.seed      = 1;
  //    params.oversampling_factor = 0;
  //
  //    raft::random::RngState rng(params.rng_state.seed, params.rng_state.type);
  //
  //    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
  //    auto labels = raft::make_device_vector<int, int>(handle, n_samples);
  //
  //    raft::random::make_blobs<T, int>(X.data_handle(),
  //                                     labels.data_handle(),
  //                                     n_samples,
  //                                     n_features,
  //                                     params.n_clusters,
  //                                     stream,
  //                                     true,
  //                                     nullptr,
  //                                     nullptr,
  //                                     T(1.0),
  //                                     false,
  //                                     (T)-10.0f,
  //                                     (T)10.0f,
  //                                     (uint64_t)1234);
  //    d_labels.resize(n_samples, stream);
  //    d_labels_ref.resize(n_samples, stream);
  //    d_centroids.resize(params.n_clusters * n_features, stream);
  //    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);
  //    rmm::device_uvector<T> d_sample_weight(n_samples, stream);
  //    thrust::fill(
  //      thrust::cuda::par.on(stream), d_sample_weight.data(), d_sample_weight.data() + n_samples,
  //      1);
  //    auto weight_view =
  //      raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples);
  //
  //    T inertia  = 0;
  //    int n_iter = 0;
  //    rmm::device_uvector<char> workspace(0, stream);
  //    rmm::device_uvector<T> L2NormBuf_OR_DistBuf(0, stream);
  //    rmm::device_uvector<T> inRankCp(0, stream);
  //    auto X_view = raft::make_const_mdspan(X.view());
  //    auto centroids_view =
  //      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
  //    auto miniX = raft::make_device_matrix<T, int>(handle, n_samples / 4, n_features);
  //
  //    // Initialize kmeans on a portion of X
  //    raft::cluster::kmeans::shuffle_and_gather(
  //      handle,
  //      X_view,
  //      raft::make_device_matrix_view<T, int>(miniX.data_handle(), miniX.extent(0),
  //      miniX.extent(1)), miniX.extent(0), params.rng_state.seed);
  //
  //    raft::cluster::kmeans::init_plus_plus(
  //      handle, params, raft::make_const_mdspan(miniX.view()), centroids_view, workspace);
  //
  //    auto minClusterDistance = raft::make_device_vector<T, int>(handle, n_samples);
  //    auto minClusterAndDistance =
  //      raft::make_device_vector<raft::KeyValuePair<int, T>, int>(handle, n_samples);
  //    auto L2NormX           = raft::make_device_vector<T, int>(handle, n_samples);
  //    auto clusterCostBefore = raft::make_device_scalar<T>(handle, 0);
  //    auto clusterCostAfter  = raft::make_device_scalar<T>(handle, 0);
  //
  //    raft::linalg::rowNorm(L2NormX.data_handle(),
  //                          X.data_handle(),
  //                          X.extent(1),
  //                          X.extent(0),
  //                          raft::linalg::L2Norm,
  //                          true,
  //                          stream);
  //
  //    raft::cluster::kmeans::min_cluster_distance(handle,
  //                                                X_view,
  //                                                centroids_view,
  //                                                minClusterDistance.view(),
  //                                                L2NormX.view(),
  //                                                L2NormBuf_OR_DistBuf,
  //                                                params.metric,
  //                                                params.batch_samples,
  //                                                params.batch_centroids,
  //                                                workspace);
  //
  //    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostBefore.view());
  //
  //    // Run a fit of kmeans
  //    raft::cluster::kmeans::fit_main(handle,
  //                                    params,
  //                                    X_view,
  //                                    weight_view,
  //                                    centroids_view,
  //                                    raft::make_host_scalar_view(&inertia),
  //                                    raft::make_host_scalar_view(&n_iter),
  //                                    workspace);
  //
  //    // Check that the cluster cost decreased
  //    raft::cluster::kmeans::min_cluster_distance(handle,
  //                                                X_view,
  //                                                centroids_view,
  //                                                minClusterDistance.view(),
  //                                                L2NormX.view(),
  //                                                L2NormBuf_OR_DistBuf,
  //                                                params.metric,
  //                                                params.batch_samples,
  //                                                params.batch_centroids,
  //                                                workspace);
  //
  //    run_cluster_cost(handle, minClusterDistance.view(), workspace, clusterCostAfter.view());
  //    T h_clusterCostBefore = T(0);
  //    T h_clusterCostAfter  = T(0);
  //    raft::update_host(&h_clusterCostBefore, clusterCostBefore.data_handle(), 1, stream);
  //    raft::update_host(&h_clusterCostAfter, clusterCostAfter.data_handle(), 1, stream);
  //    ASSERT_TRUE(h_clusterCostAfter < h_clusterCostBefore);
  //
  //    // Count samples in clusters using 2 methods and compare them
  //    // Fill minClusterAndDistance
  //    raft::cluster::kmeans::min_cluster_and_distance(
  //      handle,
  //      X_view,
  //      raft::make_device_matrix_view<const T, int>(
  //        d_centroids.data(), params.n_clusters, n_features),
  //      minClusterAndDistance.view(),
  //      L2NormX.view(),
  //      L2NormBuf_OR_DistBuf,
  //      params.metric,
  //      params.batch_samples,
  //      params.batch_centroids,
  //      workspace);
  //    raft::cluster::kmeans::KeyValueIndexOp<int, T> conversion_op;
  //    thrust::transform_iterator<raft::cluster::kmeans::KeyValueIndexOp<int, T>,
  //                               raft::KeyValuePair<int, T>*>
  //      itr(minClusterAndDistance.data_handle(), conversion_op);
  //
  //    auto sampleCountInCluster = raft::make_device_vector<T, int>(handle, params.n_clusters);
  //    auto weigthInCluster      = raft::make_device_vector<T, int>(handle, params.n_clusters);
  //    auto newCentroids = raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features);
  //    raft::cluster::kmeans::update_centroids(handle,
  //                                            X_view,
  //                                            weight_view,
  //                                            raft::make_device_matrix_view<const T, int>(
  //                                              d_centroids.data(), params.n_clusters,
  //                                              n_features),
  //                                            itr,
  //                                            weigthInCluster.view(),
  //                                            newCentroids.view());
  //    raft::cluster::kmeans::count_samples_in_cluster(handle,
  //                                                    params,
  //                                                    X_view,
  //                                                    L2NormX.view(),
  //                                                    newCentroids.view(),
  //                                                    workspace,
  //                                                    sampleCountInCluster.view());
  //
  //    ASSERT_TRUE(devArrMatch(sampleCountInCluster.data_handle(),
  //                            weigthInCluster.data_handle(),
  //                            params.n_clusters,
  //                            CompareApprox<T>(params.tol)));
  //  }

  void basicTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);
    auto stream = raft::resource::get_cuda_stream(handle);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    std::optional<raft::device_vector_view<const T, int>> d_sw = std::nullopt;
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples));
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   1);
    }

    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);

    T inertia   = 0;
    int n_iter  = 0;
    auto X_view = raft::make_const_mdspan(X.view());

    cuvs::cluster::kmeans::fit_predict(
      handle,
      params,
      X_view,
      d_sw,
      d_centroids_view,
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int>(&n_iter));

    raft::resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, raft::resource::get_cuda_stream(handle));

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override
  {
    basicTest();
    //    apiTest();
  }

 protected:
  raft::resources handle;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  double score;
  cuvs::cluster::kmeans::params params;
};

const std::vector<KmeansInputs<float>> inputsf2 = {{1000, 32, 5, 0.0001f, true},
                                                   {1000, 32, 5, 0.0001f, false},
                                                   {1000, 100, 20, 0.0001f, true},
                                                   {1000, 100, 20, 0.0001f, false},
                                                   {10000, 32, 10, 0.0001f, true},
                                                   {10000, 32, 10, 0.0001f, false},
                                                   {10000, 100, 50, 0.0001f, true},
                                                   {10000, 100, 50, 0.0001f, false},
                                                   {10000, 500, 100, 0.0001f, true},
                                                   {10000, 500, 100, 0.0001f, false}};

const std::vector<KmeansInputs<double>> inputsd2 = {{1000, 32, 5, 0.0001, true},
                                                    {1000, 32, 5, 0.0001, false},
                                                    {1000, 100, 20, 0.0001, true},
                                                    {1000, 100, 20, 0.0001, false},
                                                    {10000, 32, 10, 0.0001, true},
                                                    {10000, 32, 10, 0.0001, false},
                                                    {10000, 100, 50, 0.0001, true},
                                                    {10000, 100, 50, 0.0001, false},
                                                    {10000, 500, 100, 0.0001, true},
                                                    {10000, 500, 100, 0.0001, false}};

typedef KmeansTest<float> KmeansTestF;

TEST_P(KmeansTestF, Result) { ASSERT_TRUE(score == 1.0); }

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

// ============================================================================
// Batched KMeans Tests (fit + predict with host data)
// ============================================================================

template <typename T>
struct KmeansBatchedInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

template <typename T>
class KmeansFitBatchedTest : public ::testing::TestWithParam<KmeansBatchedInputs<T>> {
 protected:
  KmeansFitBatchedTest()
    : d_labels(0, raft::resource::get_cuda_stream(handle)),
      d_labels_ref(0, raft::resource::get_cuda_stream(handle)),
      d_centroids(0, raft::resource::get_cuda_stream(handle)),
      d_centroids_ref(0, raft::resource::get_cuda_stream(handle))
  {
  }

  void fitBatchedTest()
  {
    testparams = ::testing::TestWithParam<KmeansBatchedInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto stream = raft::resource::get_cuda_stream(handle);
    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    // Copy X to host for batched API
    std::vector<T> h_X(n_samples * n_features);
    raft::update_host(h_X.data(), X.data_handle(), n_samples * n_features, stream);
    raft::resource::sync_stream(handle, stream);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);
    d_centroids_ref.resize(params.n_clusters * n_features, stream);
    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, stream);

    raft::random::RngState rng(params.rng_state.seed);
    raft::random::uniform(
      handle, rng, d_centroids.data(), params.n_clusters * n_features, T(-1), T(1));
    raft::copy(d_centroids_ref.data(), d_centroids.data(), params.n_clusters * n_features, stream);

    auto h_X_view = raft::make_host_matrix_view<const T, int>(h_X.data(), n_samples, n_features);
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);

    // Run device fit to get reference centroids
    std::optional<raft::device_vector_view<const T, int>> d_sw = std::nullopt;
    rmm::device_uvector<T> d_sample_weight(0, stream);
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples));
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   T(1));
    }

    auto d_centroids_ref_view =
      raft::make_device_matrix_view<T, int>(d_centroids_ref.data(), params.n_clusters, n_features);

    params.init          = cuvs::cluster::kmeans::params::Array;
    params.n_init        = 1;
    params.inertia_check = true;
    params.max_iter      = 20;

    T ref_inertia  = 0;
    int ref_n_iter = 0;
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               raft::make_const_mdspan(X.view()),
                               d_sw,
                               d_centroids_ref_view,
                               raft::make_host_scalar_view<T>(&ref_inertia),
                               raft::make_host_scalar_view<int>(&ref_n_iter));

    cuvs::cluster::kmeans::params batched_params = params;
    batched_params.inertia_check                 = true;
    batched_params.streaming_batch_size                    = std::min(n_samples, 256);

    std::optional<raft::host_vector_view<const T, int>> h_sw = std::nullopt;
    std::vector<T> h_sample_weight;
    if (testparams.weighted) {
      h_sample_weight.resize(n_samples, T(1));
      h_sw = std::make_optional(
        raft::make_host_vector_view<const T, int>(h_sample_weight.data(), n_samples));
    }

    T inertia  = 0;
    int n_iter = 0;

    cuvs::cluster::kmeans::fit(handle,
                               batched_params,
                               h_X_view,
                               h_sw,
                               d_centroids_view,
                               raft::make_host_scalar_view<T>(&inertia),
                               raft::make_host_scalar_view<int>(&n_iter));

    raft::resource::sync_stream(handle, stream);

    centroids_match = devArrMatch(d_centroids_ref.data(),
                                  d_centroids.data(),
                                  params.n_clusters,
                                  n_features,
                                  CompareApprox<T>(T(1e-2)),
                                  stream);

    // Also check label quality via ARI
    T ref_pred_inertia = 0;
    cuvs::cluster::kmeans::predict(
      handle,
      params,
      raft::make_const_mdspan(X.view()),
      std::optional<raft::device_vector_view<const T, int>>(std::nullopt),
      raft::make_device_matrix_view<const T, int>(
        d_centroids_ref.data(), params.n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_ref.data(), n_samples),
      true,
      raft::make_host_scalar_view<T>(&ref_pred_inertia));

    // Also check label quality via ARI
    T pred_inertia = 0;
    cuvs::cluster::kmeans::predict(
      handle,
      params,
      raft::make_const_mdspan(X.view()),
      std::optional<raft::device_vector_view<const T, int>>(std::nullopt),
      raft::make_device_matrix_view<const T, int>(
        d_centroids.data(), params.n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples),
      true,
      raft::make_host_scalar_view<T>(&pred_inertia));

    raft::resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, raft::resource::get_cuda_stream(handle));

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream);
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream);
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }
  }

  void SetUp() override { fitBatchedTest(); }

 protected:
  raft::resources handle;
  KmeansBatchedInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_centroids_ref;
  double score;
  testing::AssertionResult centroids_match = testing::AssertionSuccess();
  cuvs::cluster::kmeans::params params;
};

template <typename T>
class KmeansPredictBatchedTest : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansPredictBatchedTest()
    : d_labels(0, raft::resource::get_cuda_stream(handle)),
      d_labels_ref(0, raft::resource::get_cuda_stream(handle)),
      d_centroids(0, raft::resource::get_cuda_stream(handle)),
      d_sample_weight(0, raft::resource::get_cuda_stream(handle))
  {
  }

  void predictBatchedTest()
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto stream = raft::resource::get_cuda_stream(handle);
    auto X      = raft::make_device_matrix<T, int>(handle, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(handle, n_samples);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    // Fit on device to get centroids
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);

    std::optional<raft::device_vector_view<const T, int>> d_sw = std::nullopt;
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples));
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   T(1));
    }

    T fit_inertia  = 0;
    int fit_n_iter = 0;
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               raft::make_const_mdspan(X.view()),
                               d_sw,
                               d_centroids_view,
                               raft::make_host_scalar_view<T>(&fit_inertia),
                               raft::make_host_scalar_view<int>(&fit_n_iter));

    T ref_inertia = 0;
    cuvs::cluster::kmeans::predict(
      handle,
      params,
      raft::make_const_mdspan(X.view()),
      std::optional<raft::device_vector_view<const T, int>>(std::nullopt),
      raft::make_device_matrix_view<const T, int>(
        d_centroids.data(), params.n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_ref.data(), n_samples),
      true,
      raft::make_host_scalar_view<T>(&ref_inertia));

    std::vector<T> h_X(n_samples * n_features);
    raft::update_host(h_X.data(), X.data_handle(), n_samples * n_features, stream);
    raft::resource::sync_stream(handle, stream);

    auto h_X_view = raft::make_host_matrix_view<const T, int64_t>(
      h_X.data(), (int64_t)n_samples, (int64_t)n_features);
    auto centroids_const_view = raft::make_device_matrix_view<const T, int64_t>(
      d_centroids.data(), (int64_t)params.n_clusters, (int64_t)n_features);

    std::vector<int64_t> h_labels(n_samples);
    auto h_labels_view =
      raft::make_host_vector_view<int64_t, int64_t>(h_labels.data(), (int64_t)n_samples);

    T pred_inertia    = 0;
    params.streaming_batch_size = std::min((int64_t)n_samples, (int64_t)256);

    cuvs::cluster::kmeans::predict(
      handle,
      params,
      h_X_view,
      std::optional<raft::host_vector_view<const T, int64_t>>(std::nullopt),
      centroids_const_view,
      h_labels_view,
      true,
      raft::make_host_scalar_view<T>(&pred_inertia));

    raft::resource::sync_stream(handle, stream);

    std::vector<int> h_labels_int(n_samples);
    for (int i = 0; i < n_samples; ++i) {
      h_labels_int[i] = static_cast<int>(h_labels[i]);
    }
    raft::update_device(d_labels.data(), h_labels_int.data(), n_samples, stream);

    // Compare labels directly: batched predict should produce exact same labels
    // as device predict given the same centroids
    labels_match =
      devArrMatch(d_labels_ref.data(), d_labels.data(), n_samples, Compare<int>(), stream);
  }

  void SetUp() override { predictBatchedTest(); }

 protected:
  raft::resources handle;
  KmeansInputs<T> testparams;
  rmm::device_uvector<int> d_labels;
  rmm::device_uvector<int> d_labels_ref;
  rmm::device_uvector<T> d_centroids;
  rmm::device_uvector<T> d_sample_weight;
  testing::AssertionResult labels_match = testing::AssertionSuccess();
  cuvs::cluster::kmeans::params params;
};

// ============================================================================
// Test inputs for batched tests
// ============================================================================

const std::vector<KmeansBatchedInputs<float>> batched_inputsf2 = {
  {1000, 32, 5, 0.0001f, true},
  {1000, 32, 5, 0.0001f, false},
  {1000, 100, 20, 0.0001f, true},
  {1000, 100, 20, 0.0001f, false},
  {10000, 32, 10, 0.0001f, true},
  {10000, 32, 10, 0.0001f, false},
};

const std::vector<KmeansBatchedInputs<double>> batched_inputsd2 = {
  {1000, 32, 5, 0.0001, true},
  {1000, 32, 5, 0.0001, false},
  {1000, 100, 20, 0.0001, true},
  {1000, 100, 20, 0.0001, false},
  {10000, 32, 10, 0.0001, true},
  {10000, 32, 10, 0.0001, false},
};

// ============================================================================
// fit (host/batched) tests
// ============================================================================
typedef KmeansFitBatchedTest<float> KmeansFitBatchedTestF;
typedef KmeansFitBatchedTest<double> KmeansFitBatchedTestD;

TEST_P(KmeansFitBatchedTestF, Result)
{
  ASSERT_TRUE(centroids_match);
  ASSERT_TRUE(score == 1.0);
}

TEST_P(KmeansFitBatchedTestD, Result)
{
  ASSERT_TRUE(centroids_match);
  ASSERT_TRUE(score == 1.0);
}

INSTANTIATE_TEST_CASE_P(KmeansFitBatchedTests,
                        KmeansFitBatchedTestF,
                        ::testing::ValuesIn(batched_inputsf2));
INSTANTIATE_TEST_CASE_P(KmeansFitBatchedTests,
                        KmeansFitBatchedTestD,
                        ::testing::ValuesIn(batched_inputsd2));

// ============================================================================
// predict (host/batched) tests
// ============================================================================
typedef KmeansPredictBatchedTest<float> KmeansPredictBatchedTestF;
typedef KmeansPredictBatchedTest<double> KmeansPredictBatchedTestD;

TEST_P(KmeansPredictBatchedTestF, Result) { ASSERT_TRUE(labels_match); }
TEST_P(KmeansPredictBatchedTestD, Result) { ASSERT_TRUE(labels_match); }

INSTANTIATE_TEST_CASE_P(KmeansPredictBatchedTests,
                        KmeansPredictBatchedTestF,
                        ::testing::ValuesIn(inputsf2));
INSTANTIATE_TEST_CASE_P(KmeansPredictBatchedTests,
                        KmeansPredictBatchedTestD,
                        ::testing::ValuesIn(inputsd2));

}  // namespace cuvs
