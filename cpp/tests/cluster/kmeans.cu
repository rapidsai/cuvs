/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/cluster/detail/kmeans_batch_loader.cuh"
#include "../../src/cluster/detail/kmeans_common.cuh"
#include "../test_utils.cuh"
#include "kmeans_test_blobs.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
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

    auto stream = raft::resource::get_cuda_stream(handle);
    auto bi     = make_kmeans_blob_inputs<T>(
      handle, n_samples, n_features, params.n_clusters, /* with_host_mirror */ false);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    std::optional<raft::device_vector_view<const T, int>> d_sw = std::nullopt;
    auto d_centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      auto d_sw_view = raft::make_device_vector_view<T, int>(d_sample_weight.data(), n_samples);
      fill_kmeans_test_weights(handle, d_sw_view, kmeans_weight_mode::uniform);
      d_sw = std::make_optional(
        raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples));
    }

    raft::copy(d_labels_ref.data(), bi.d_labels_ref.data_handle(), n_samples, stream);

    T inertia   = 0;
    int n_iter  = 0;
    auto X_view = raft::make_const_mdspan(bi.d_X.view());

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

    score = raft::stats::adjusted_rand_index(d_labels_ref.data(),
                                             d_labels.data(),
                                             n_samples,
                                             raft::resource::get_cuda_stream(handle).get());

    if (score < 1.0) {
      std::stringstream ss;
      ss << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream.get());
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream.get());
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
  int device_buffer_samples;
};

template <typename T>
class KmeansFitBatchedTest : public ::testing::TestWithParam<KmeansBatchedInputs<T>> {
 protected:
  KmeansFitBatchedTest() = default;

  void prepareBlobInputs()
  {
    testparams     = ::testing::TestWithParam<KmeansBatchedInputs<T>>::GetParam();
    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;

    auto bi = make_kmeans_blob_inputs<T>(handle, n_samples, n_features, n_clusters);
    d_X.emplace(std::move(bi.d_X));
    d_labels_ref.emplace(std::move(bi.d_labels_ref));
    h_X = std::move(bi.h_X);

    if (testparams.weighted) {
      d_sample_weight.emplace(raft::make_device_vector<T, int>(handle, n_samples));
      fill_kmeans_test_weights(handle, d_sample_weight->view(), kmeans_weight_mode::uniform);
    } else {
      d_sample_weight.reset();
    }
  }

  std::optional<raft::device_vector_view<const T, int>> d_sw_view() const
  {
    if (!d_sample_weight.has_value()) return std::nullopt;
    return std::make_optional(raft::make_const_mdspan(d_sample_weight->view()));
  }

  void fitBatchedTest()
  {
    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 0;

    auto stream = raft::resource::get_cuda_stream(handle);

    d_labels.emplace(raft::make_device_vector<int, int>(handle, n_samples));
    d_centroids.emplace(raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features));
    d_centroids_ref.emplace(
      raft::make_device_matrix<T, int>(handle, params.n_clusters, n_features));

    raft::random::RngState rng(params.rng_state.seed);
    raft::random::uniform(
      handle, rng, d_centroids->data_handle(), params.n_clusters * n_features, T(-1), T(1));
    raft::copy(d_centroids_ref->data_handle(),
               d_centroids->data_handle(),
               params.n_clusters * n_features,
               stream);

    auto d_centroids_view = raft::make_device_matrix_view<T, int64_t>(
      d_centroids->data_handle(), params.n_clusters, n_features);

    auto d_sw = d_sw_view();

    params.init     = cuvs::cluster::kmeans::params::Array;
    params.max_iter = 20;

    T ref_inertia  = 0;
    int ref_n_iter = 0;
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               raft::make_const_mdspan(d_X->view()),
                               d_sw,
                               d_centroids_ref->view(),
                               raft::make_host_scalar_view<T>(&ref_inertia),
                               raft::make_host_scalar_view<int>(&ref_n_iter));

    cuvs::cluster::kmeans::params batched_params = params;
    batched_params.device_buffer_samples         = testparams.device_buffer_samples;

    std::optional<raft::host_vector_view<const T, int64_t>> h_sw = std::nullopt;
    auto h_sample_weight = raft::make_host_vector<T, int64_t>(testparams.weighted ? n_samples : 0);
    if (testparams.weighted) {
      std::fill_n(h_sample_weight.data_handle(), n_samples, T(1));
      h_sw = std::make_optional(raft::make_const_mdspan(h_sample_weight.view()));
    }

    T inertia      = 0;
    int64_t n_iter = 0;

    cuvs::cluster::kmeans::fit(handle,
                               batched_params,
                               raft::make_const_mdspan(h_X->view()),
                               h_sw,
                               d_centroids_view,
                               raft::make_host_scalar_view<T>(&inertia),
                               raft::make_host_scalar_view<int64_t>(&n_iter));

    raft::resource::sync_stream(handle, stream);

    centroids_match = devArrMatch(d_centroids_ref->data_handle(),
                                  d_centroids->data_handle(),
                                  params.n_clusters,
                                  n_features,
                                  CompareApprox<T>(T(1e-2)),
                                  stream.get());

    T ref_pred_inertia = 0;
    cuvs::cluster::kmeans::predict(handle,
                                   params,
                                   raft::make_const_mdspan(d_X->view()),
                                   d_sw,
                                   raft::make_const_mdspan(d_centroids_ref->view()),
                                   d_labels_ref->view(),
                                   true,
                                   raft::make_host_scalar_view<T>(&ref_pred_inertia));

    T pred_inertia = 0;
    cuvs::cluster::kmeans::predict(handle,
                                   params,
                                   raft::make_const_mdspan(d_X->view()),
                                   d_sw,
                                   raft::make_const_mdspan(d_centroids->view()),
                                   d_labels->view(),
                                   true,
                                   raft::make_host_scalar_view<T>(&pred_inertia));

    raft::resource::sync_stream(handle, stream);

    score = raft::stats::adjusted_rand_index(d_labels_ref->data_handle(),
                                             d_labels->data_handle(),
                                             n_samples,
                                             raft::resource::get_cuda_stream(handle).get());

    if (score < 0.99) {
      std::stringstream ss;
      ss << "Expected: "
         << raft::arr2Str(d_labels_ref->data_handle(), 25, "d_labels_ref", stream.get());
      std::cout << (ss.str().c_str()) << '\n';
      ss.str(std::string());
      ss << "Actual: " << raft::arr2Str(d_labels->data_handle(), 25, "d_labels", stream.get());
      std::cout << (ss.str().c_str()) << '\n';
      std::cout << "Score = " << score << '\n';
    }

    inertia_match = (std::abs(ref_inertia - inertia) < T(1e-2) * std::abs(ref_inertia));

    if (!inertia_match) {
      std::cout << "Inertia mismatch: ref=" << ref_inertia << " batched=" << inertia << '\n';
    }
  }

  T fitHostWithInitSize(int64_t init_size_value)
  {
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;
    auto stream    = raft::resource::get_cuda_stream(handle);

    cuvs::cluster::kmeans::params p;
    p.n_clusters            = n_clusters;
    p.tol                   = testparams.tol;
    p.n_init                = 3;
    p.init                  = cuvs::cluster::kmeans::params::KMeansPlusPlus;
    p.max_iter              = 20;
    p.rng_state.seed        = 1;
    p.oversampling_factor   = 0;
    p.device_buffer_samples = testparams.device_buffer_samples;
    p.init_size             = init_size_value;

    auto d_centroids_buf = raft::make_device_matrix<T, int64_t>(handle, n_clusters, n_features);
    T inertia            = 0;
    int64_t n_iter       = 0;
    cuvs::cluster::kmeans::fit(
      handle,
      p,
      raft::make_const_mdspan(h_X->view()),
      std::optional<raft::host_vector_view<const T, int64_t>>{std::nullopt},
      d_centroids_buf.view(),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int64_t>(&n_iter));
    raft::resource::sync_stream(handle, stream);
    return inertia;
  }

  void runInitSizeCompare()
  {
    int n_samples         = testparams.n_row;
    int n_clusters        = testparams.n_clusters;
    int default_init_size = std::min(3 * n_clusters, n_samples);

    T inertia_default  = fitHostWithInitSize(0);
    T inertia_explicit = fitHostWithInitSize(default_init_size);
    T inertia_full     = fitHostWithInitSize(n_samples);

    ASSERT_TRUE(std::isfinite(inertia_default));
    ASSERT_TRUE(std::isfinite(inertia_explicit));
    ASSERT_TRUE(std::isfinite(inertia_full));
    ASSERT_GT(inertia_default, T(0));
    ASSERT_GT(inertia_explicit, T(0));
    ASSERT_GT(inertia_full, T(0));

    // cuTile's TF32 assignment path can introduce small FP32 convergence variation between
    // otherwise equivalent runs; keep the original tighter tolerance for double precision.
    const T rel = std::is_same_v<T, float> ? T(1e-4) : T(1e-5);

    // init_size = 0 must resolve to the documented default (min(3*k, n));
    // feeding that value explicitly should reproduce the same inertia.
    ASSERT_NEAR(inertia_default, inertia_explicit, std::abs(inertia_default) * rel);

    // Full-dataset seeding has at least as much information as the subsample
    // default, so the converged inertia should not be worse.
    ASSERT_LE(inertia_full, inertia_default * (T(1) + rel));
  }

  T fitKMeansPlusPlus(int n_init_value)
  {
    int n_features = testparams.n_col;
    int n_clusters = testparams.n_clusters;
    auto stream    = raft::resource::get_cuda_stream(handle);

    cuvs::cluster::kmeans::params p;
    p.n_clusters          = n_clusters;
    p.tol                 = testparams.tol;
    p.n_init              = n_init_value;
    p.init                = cuvs::cluster::kmeans::params::KMeansPlusPlus;
    p.max_iter            = 10;
    p.rng_state.seed      = 7;
    p.oversampling_factor = 0;

    auto d_centroids_buf = raft::make_device_matrix<T, int>(handle, n_clusters, n_features);
    T inertia            = 0;
    int n_iter           = 0;
    cuvs::cluster::kmeans::fit(handle,
                               p,
                               raft::make_const_mdspan(d_X->view()),
                               std::optional<raft::device_vector_view<const T, int>>{std::nullopt},
                               d_centroids_buf.view(),
                               raft::make_host_scalar_view<T>(&inertia),
                               raft::make_host_scalar_view<int>(&n_iter));
    raft::resource::sync_stream(handle, stream);
    return inertia;
  }

  void runMultiSeedCheck()
  {
    T inertia1 = fitKMeansPlusPlus(1);
    T inertia3 = fitKMeansPlusPlus(3);
    ASSERT_GT(inertia1, T(0));
    ASSERT_GT(inertia3, T(0));
    // n_init > 1 keeps the best trial, so it should not be worse than
    // n_init == 1.
    const T rel = T(1e-5);
    ASSERT_LE(inertia3, inertia1 * (T(1) + rel));
  }

  void runZeroCost()
  {
    int n_samples  = testparams.n_row;
    int n_features = testparams.n_col;
    int n_clusters = n_samples;
    auto stream    = raft::resource::get_cuda_stream(handle);

    auto d_centroids_buf = raft::make_device_matrix<T, int>(handle, n_clusters, n_features);
    raft::copy(d_centroids_buf.data_handle(),
               d_X->data_handle(),
               static_cast<size_t>(n_samples) * n_features,
               stream);

    cuvs::cluster::kmeans::params p;
    p.n_clusters          = n_clusters;
    p.tol                 = testparams.tol;
    p.n_init              = 1;
    p.init                = cuvs::cluster::kmeans::params::Array;
    p.max_iter            = 5;
    p.rng_state.seed      = 1;
    p.oversampling_factor = 0;

    T inertia  = 0;
    int n_iter = 0;
    ASSERT_NO_THROW(cuvs::cluster::kmeans::fit(
      handle,
      p,
      raft::make_const_mdspan(d_X->view()),
      std::optional<raft::device_vector_view<const T, int>>{std::nullopt},
      d_centroids_buf.view(),
      raft::make_host_scalar_view<T>(&inertia),
      raft::make_host_scalar_view<int>(&n_iter)));
    raft::resource::sync_stream(handle, stream);

    // The expanded L2 formula ||x||^2 - 2 x.c + ||c||^2 does not cancel to
    // exactly 0 even when x == c due to float roundoff. The largest residual
    // inertia observed across parameterized blob shapes is ~27.95; use an
    // absolute upper bound of 100 for headroom.
    ASSERT_LE(inertia, T(100));
  }

 protected:
  raft::resources handle;
  KmeansBatchedInputs<T> testparams;
  std::optional<raft::device_vector<int, int>> d_labels;
  std::optional<raft::device_vector<int, int>> d_labels_ref;
  std::optional<raft::device_matrix<T, int>> d_centroids;
  std::optional<raft::device_matrix<T, int>> d_centroids_ref;
  std::optional<raft::device_matrix<T, int>> d_X;
  std::optional<raft::device_vector<T, int>> d_sample_weight;
  std::optional<raft::host_matrix<T, int64_t>> h_X;
  double score;
  testing::AssertionResult centroids_match = testing::AssertionSuccess();
  bool inertia_match                       = false;
  cuvs::cluster::kmeans::params params;
};

// ============================================================================
// Test inputs for batched tests
// ============================================================================

const std::vector<KmeansBatchedInputs<float>> batched_inputsf2 = {
  {1000, 32, 5, 0.0001f, true, 256},
  {1000, 64, 5, 0.0001f, false, 500},
  {1000, 100, 20, 0.0001f, true, 30},
  {1000, 10, 20, 0.0001f, false, 30},
  {10000, 16, 10, 0.00001f, true, 1000},
  {10000, 96, 10, 0.0001f, false, 10000},
};

const std::vector<KmeansBatchedInputs<double>> batched_inputsd2 = {
  {1000, 32, 5, 0.0001, true, 256},
  {1000, 64, 5, 0.0001, false, 500},
  {1000, 100, 20, 0.0001, true, 30},
  {1000, 10, 20, 0.0001, false, 30},
  {10000, 16, 10, 0.0001, true, 1000},
  {10000, 96, 10, 0.0001, false, 10000},
};

// ============================================================================
// fit (host/batched) tests
// ============================================================================
typedef KmeansFitBatchedTest<float> KmeansFitBatchedTestF;
typedef KmeansFitBatchedTest<double> KmeansFitBatchedTestD;

TEST_P(KmeansFitBatchedTestF, Result)
{
  prepareBlobInputs();
  fitBatchedTest();
  // AUTO may select cuTile, whose TF32 assignment arithmetic can converge to slightly different
  // centroid values when accumulation is split into outer host batches. Equivalent assignments and
  // clustering cost are the stable behavioral contract.
  ASSERT_TRUE(score >= 0.99);
  ASSERT_TRUE(inertia_match);
  runInitSizeCompare();
  runMultiSeedCheck();
  runZeroCost();
}

TEST_P(KmeansFitBatchedTestD, Result)
{
  prepareBlobInputs();
  fitBatchedTest();
  ASSERT_TRUE(centroids_match);
  ASSERT_TRUE(score >= 0.99);
  ASSERT_TRUE(inertia_match);
  runInitSizeCompare();
  runMultiSeedCheck();
  runZeroCost();
}

INSTANTIATE_TEST_CASE_P(KmeansFitBatchedTests,
                        KmeansFitBatchedTestF,
                        ::testing::ValuesIn(batched_inputsf2));
INSTANTIATE_TEST_CASE_P(KmeansFitBatchedTests,
                        KmeansFitBatchedTestD,
                        ::testing::ValuesIn(batched_inputsd2));

// ============================================================================
// Byte-input KMeans fit tests
// ============================================================================

namespace {

struct byte_fit_result {
  std::vector<float> centroids;
  float inertia{};
  int64_t n_iter{};
};

template <typename ByteT>
constexpr float byte_divisor()
{
  static_assert(std::is_same_v<ByteT, int8_t> || std::is_same_v<ByteT, uint8_t>);
  return std::is_same_v<ByteT, int8_t> ? 128.0f : 256.0f;
}

template <typename ByteT>
struct byte_blob_quantizer {
  RAFT_INLINE_FUNCTION ByteT operator()(float value) const
  {
    constexpr int offset = std::is_same_v<ByteT, uint8_t> ? 128 : 0;
    return static_cast<ByteT>(static_cast<int>(value * 42.0f) + offset);
  }
};

template <typename ByteT>
class KmeansByteFitTest : public ::testing::Test {
 protected:
  static constexpr int64_t n_samples  = 11;
  static constexpr int64_t n_features = 3;
  static constexpr int64_t n_clusters = 2;
  static constexpr int64_t batch_rows = 4;

  void SetUp() override { generate_inputs(n_samples); }

  void generate_inputs(int64_t rows)
  {
    auto stream = raft::resource::get_cuda_stream(handle);
    auto blobs  = raft::make_device_matrix<float, int64_t>(handle, rows, n_features);
    auto labels = raft::make_device_vector<int64_t, int64_t>(handle, rows);
    auto bytes  = raft::make_device_matrix<ByteT, int64_t>(handle, rows, n_features);

    // Match the balanced k-means byte tests: seeded float blobs, then scale and quantize.
    raft::random::make_blobs<float, int64_t>(blobs.data_handle(),
                                             labels.data_handle(),
                                             rows,
                                             n_features,
                                             n_clusters,
                                             stream.get(),
                                             true,
                                             nullptr,
                                             nullptr,
                                             0.1f,
                                             true,
                                             -1.0f,
                                             1.0f,
                                             uint64_t{1234});
    raft::linalg::unaryOp(bytes.data_handle(),
                          blobs.data_handle(),
                          rows * n_features,
                          byte_blob_quantizer<ByteT>{},
                          stream.get());

    h_byte_input.resize(rows * n_features);
    std::vector<int64_t> h_labels(rows);
    raft::copy(h_byte_input.data(), bytes.data_handle(), h_byte_input.size(), stream);
    raft::copy(h_labels.data(), labels.data_handle(), h_labels.size(), stream);
    raft::resource::sync_stream(handle);

    // Use the quantized values for the float reference, including initial centroids.
    h_float_input.resize(h_byte_input.size());
    std::transform(
      h_byte_input.begin(), h_byte_input.end(), h_float_input.begin(), [](ByteT value) {
        return static_cast<float>(value) / byte_divisor<ByteT>();
      });
    h_initial_centroids.resize(n_clusters * n_features);
    for (int64_t cluster = 0; cluster < n_clusters; ++cluster) {
      auto row = std::find(h_labels.begin(), h_labels.end(), cluster);
      ASSERT_NE(row, h_labels.end());
      std::copy_n(h_float_input.begin() + (row - h_labels.begin()) * n_features,
                  n_features,
                  h_initial_centroids.begin() + cluster * n_features);
    }

    // Nonuniform fractional weights have a period distinct from either batch size.
    h_weights.resize(rows);
    for (int64_t row = 0; row < rows; ++row) {
      h_weights[row] = 0.5f + 0.25f * static_cast<float>(row % 17);
    }
    if constexpr (std::is_same_v<ByteT, int8_t>) {
      EXPECT_LT(*std::min_element(h_byte_input.begin(), h_byte_input.end()), ByteT{0});
    } else {
      EXPECT_GT(*std::max_element(h_byte_input.begin(), h_byte_input.end()),
                static_cast<ByteT>(std::numeric_limits<int8_t>::max()));
    }
  }

  cuvs::cluster::kmeans::params make_params(cuvs::cluster::kmeans::params::InitMethod init) const
  {
    cuvs::cluster::kmeans::params params;
    params.n_clusters            = n_clusters;
    params.init                  = init;
    params.max_iter              = 30;
    params.tol                   = 1e-7;
    params.n_init                = 1;
    params.rng_state.seed        = 73;
    params.oversampling_factor   = 0;
    params.device_buffer_samples = batch_rows;
    return params;
  }

  template <typename InputT>
  byte_fit_result fit_host(const std::vector<InputT>& input,
                           const cuvs::cluster::kmeans::params& params,
                           bool weighted)
  {
    const auto n_samples = static_cast<int64_t>(input.size()) / n_features;
    auto stream          = raft::resource::get_cuda_stream(handle);
    auto d_centroids     = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
    raft::copy(
      d_centroids.data_handle(), h_initial_centroids.data(), h_initial_centroids.size(), stream);

    std::optional<raft::host_vector_view<const float, int64_t>> weights = std::nullopt;
    if (weighted) {
      weights = raft::make_host_vector_view<const float, int64_t>(h_weights.data(), n_samples);
    }

    byte_fit_result result;
    cuvs::cluster::kmeans::fit(
      handle,
      params,
      raft::make_host_matrix_view<const InputT, int64_t>(input.data(), n_samples, n_features),
      weights,
      d_centroids.view(),
      raft::make_host_scalar_view<float>(&result.inertia),
      raft::make_host_scalar_view<int64_t>(&result.n_iter));

    result.centroids.resize(h_initial_centroids.size());
    raft::copy(result.centroids.data(), d_centroids.data_handle(), result.centroids.size(), stream);
    raft::resource::sync_stream(handle);
    return result;
  }

  byte_fit_result fit_byte_device(const cuvs::cluster::kmeans::params& params, bool weighted)
  {
    const auto n_samples = static_cast<int64_t>(h_byte_input.size()) / n_features;
    auto stream          = raft::resource::get_cuda_stream(handle);
    auto d_input         = raft::make_device_matrix<ByteT, int64_t>(handle, n_samples, n_features);
    auto d_weights   = raft::make_device_vector<float, int64_t>(handle, weighted ? n_samples : 0);
    auto d_centroids = raft::make_device_matrix<float, int64_t>(handle, n_clusters, n_features);
    raft::copy(d_input.data_handle(), h_byte_input.data(), h_byte_input.size(), stream);
    raft::copy(
      d_centroids.data_handle(), h_initial_centroids.data(), h_initial_centroids.size(), stream);

    std::optional<raft::device_vector_view<const float, int64_t>> weights = std::nullopt;
    if (weighted) {
      raft::copy(d_weights.data_handle(), h_weights.data(), h_weights.size(), stream);
      weights = raft::make_const_mdspan(d_weights.view());
    }

    byte_fit_result result;
    cuvs::cluster::kmeans::fit(handle,
                               params,
                               raft::make_const_mdspan(d_input.view()),
                               weights,
                               d_centroids.view(),
                               raft::make_host_scalar_view<float>(&result.inertia),
                               raft::make_host_scalar_view<int64_t>(&result.n_iter));

    result.centroids.resize(h_initial_centroids.size());
    raft::copy(result.centroids.data(), d_centroids.data_handle(), result.centroids.size(), stream);
    raft::resource::sync_stream(handle);
    return result;
  }

  void expect_result_near(const byte_fit_result& expected, const byte_fit_result& actual)
  {
    ASSERT_EQ(expected.centroids.size(), actual.centroids.size());
    auto expected_centroids = expected.centroids;
    auto actual_centroids   = actual.centroids;
    if (expected_centroids[0] > expected_centroids[n_features]) {
      std::swap_ranges(expected_centroids.begin(),
                       expected_centroids.begin() + n_features,
                       expected_centroids.begin() + n_features);
    }
    if (actual_centroids[0] > actual_centroids[n_features]) {
      std::swap_ranges(actual_centroids.begin(),
                       actual_centroids.begin() + n_features,
                       actual_centroids.begin() + n_features);
    }
    for (std::size_t i = 0; i < expected_centroids.size(); ++i) {
      EXPECT_NEAR(actual_centroids[i], expected_centroids[i], 2e-5f) << "centroid element " << i;
    }
    EXPECT_NEAR(
      actual.inertia, expected.inertia, std::max(2e-6f, std::abs(expected.inertia) * 2e-4f));
  }

  bool has_fractional_byte_coordinate(const byte_fit_result& result) const
  {
    return std::any_of(result.centroids.begin(), result.centroids.end(), [](float value) {
      const float byte_coordinate = value * byte_divisor<ByteT>();
      return std::abs(byte_coordinate - std::round(byte_coordinate)) > 1e-3f;
    });
  }

  raft::resources handle;
  std::vector<ByteT> h_byte_input;
  std::vector<float> h_float_input;
  std::vector<float> h_weights;
  std::vector<float> h_initial_centroids;
};

using KmeansByteTypes = ::testing::Types<int8_t, uint8_t>;
TYPED_TEST_SUITE(KmeansByteFitTest, KmeansByteTypes);

TYPED_TEST(KmeansByteFitTest, ArrayHostAndDeviceMatchNormalizedFloat)
{
  auto params = this->make_params(cuvs::cluster::kmeans::params::Array);
  for (bool weighted : {false, true}) {
    SCOPED_TRACE(weighted ? "nonuniform weights" : "unweighted");
    const auto expected = this->fit_host(this->h_float_input, params, weighted);
    const auto host     = this->fit_host(this->h_byte_input, params, weighted);
    const auto device   = this->fit_byte_device(params, weighted);
    this->expect_result_near(expected, host);
    this->expect_result_near(expected, device);
    if (weighted) {
      EXPECT_TRUE(this->has_fractional_byte_coordinate(host));
      EXPECT_TRUE(this->has_fractional_byte_coordinate(device));
    }
  }
}

TYPED_TEST(KmeansByteFitTest, FixedSeedInitializersMatchNormalizedFloat)
{
  for (auto init :
       {cuvs::cluster::kmeans::params::Random, cuvs::cluster::kmeans::params::KMeansPlusPlus}) {
    SCOPED_TRACE(init == cuvs::cluster::kmeans::params::Random ? "Random" : "KMeansPlusPlus");
    auto params = this->make_params(init);
    // Stop after seeding so the comparison validates the sampled and mapped
    // initialization rows rather than only the converged solution.
    params.max_iter = 0;
    // An explicit init_size may exceed the steady-state batch while remaining bounded by n_samples.
    if (init == cuvs::cluster::kmeans::params::KMeansPlusPlus) { params.init_size = 7; }
    const auto expected = this->fit_host(this->h_float_input, params, false);
    const auto host     = this->fit_host(this->h_byte_input, params, false);
    const auto device   = this->fit_byte_device(params, false);
    this->expect_result_near(expected, host);
    this->expect_result_near(expected, device);
  }
}

TYPED_TEST(KmeansByteFitTest, BatchSizingMatchesNormalizedFloat)
{
  for (int64_t outer_rows : {TestFixture::batch_rows, int64_t{0}}) {
    SCOPED_TRACE(outer_rows);
    auto default_params = this->make_params(cuvs::cluster::kmeans::params::KMeansPlusPlus);
    default_params.device_buffer_samples = outer_rows;
    default_params.max_iter              = 0;
    auto explicit_params                 = default_params;
    explicit_params.init_size =
      std::min<int64_t>({3 * TestFixture::n_clusters,
                         TestFixture::n_samples,
                         outer_rows > 0 ? outer_rows : TestFixture::n_samples});

    const auto expected_host = this->fit_host(this->h_float_input, explicit_params, false);
    const auto host          = this->fit_host(this->h_byte_input, default_params, false);
    this->expect_result_near(expected_host, host);

    // Device input ignores the host staging size; all rows are available for conversion.
    explicit_params.init_size =
      std::min<int64_t>(3 * TestFixture::n_clusters, TestFixture::n_samples);
    const auto expected_device = this->fit_host(this->h_float_input, explicit_params, false);
    const auto device          = this->fit_byte_device(default_params, false);
    this->expect_result_near(expected_device, device);
  }
}

TYPED_TEST(KmeansByteFitTest, DefaultInitializationRetainsEnoughRowsForClusters)
{
  auto params                  = this->make_params(cuvs::cluster::kmeans::params::KMeansPlusPlus);
  params.device_buffer_samples = 0;
  params.max_iter              = 0;
  params.init_size             = TestFixture::n_clusters;
  // Scalable seeding's random fallback uses std::rand rather than params.rng_state.
  // Reset it before each fit so reference and byte runs choose the same fallback rows.
  const auto fallback_seed = static_cast<unsigned int>(params.rng_state.seed);
  std::vector<byte_fit_result> expected;
  for (double oversampling_factor : {0.0, 2.0}) {
    params.oversampling_factor = oversampling_factor;
    std::srand(fallback_seed);
    expected.push_back(this->fit_host(this->h_float_input, params, false));
  }

  // This limit makes conversion use one row, but seeding still needs at least K rows.
  // In particular, scalable seeding must not gather more random centers than sampled rows.
  raft::resource::set_workspace_to_global_resource(this->handle, std::size_t{128});
  params.init_size = 0;
  for (int scalable = 0; scalable < 2; ++scalable) {
    SCOPED_TRACE(scalable ? "scalable KMeansPlusPlus" : "classic KMeansPlusPlus");
    params.oversampling_factor = scalable ? 2.0 : 0.0;
    std::srand(fallback_seed);
    const auto host = this->fit_host(this->h_byte_input, params, false);
    std::srand(fallback_seed);
    const auto device = this->fit_byte_device(params, false);
    this->expect_result_near(expected[scalable], host);
    this->expect_result_near(expected[scalable], device);
  }
}

TYPED_TEST(KmeansByteFitTest, ConversionBudgetIsIndependentOfHostStagingSize)
{
  constexpr int64_t n_rows              = 269;
  constexpr int64_t outer_rows          = 133;
  constexpr int64_t conversion_rows     = 64;
  constexpr std::size_t workspace_limit = 8192;

  this->generate_inputs(n_rows);

  auto params                  = this->make_params(cuvs::cluster::kmeans::params::Array);
  params.device_buffer_samples = outer_rows;
  std::vector<byte_fit_result> expected;
  for (bool weighted : {false, true}) {
    expected.push_back(this->fit_host(this->h_float_input, params, weighted));
  }

  // An 8 KiB workspace limit rounds conversion down to 64 rows. A 133-row host buffer
  // therefore contains two full conversion batches and a five-row tail; the final outer
  // batch contains three rows. Zero instead stages all 269 byte rows in one buffer.
  raft::resource::set_workspace_to_global_resource(this->handle, workspace_limit);
  for (bool prefetch : {false, true}) {
    SCOPED_TRACE(prefetch ? "prefetch" : "single staging buffer");
    if (prefetch) {
      raft::resource::set_cuda_stream_pool(this->handle,
                                           std::make_shared<rmm::cuda_stream_pool>(1));
    }
    for (bool weighted : {false, true}) {
      SCOPED_TRACE(weighted ? "nonuniform weights" : "unweighted");
      for (int64_t buffer_rows : {outer_rows, int64_t{0}}) {
        SCOPED_TRACE(buffer_rows);
        params.device_buffer_samples = buffer_rows;
        rmm::mr::statistics_resource_adaptor allocations{
          rmm::mr::get_current_device_resource_ref()};
        raft::resource::set_large_workspace_resource(this->handle,
                                                     raft::mr::device_resource{allocations});

        const auto host = this->fit_host(this->h_byte_input, params, weighted);
        this->expect_result_near(expected[weighted], host);
        const auto staging_rows  = buffer_rows == 0 ? n_rows : buffer_rows;
        const auto staging_slots = prefetch && staging_rows < n_rows ? 2 : 1;
        const auto staging_bytes =
          staging_slots * staging_rows *
          (TestFixture::n_features * sizeof(TypeParam) + (weighted ? sizeof(float) : 0));
        constexpr auto conversion_bytes = conversion_rows * TestFixture::n_features * sizeof(float);
        // Check actual allocations, so a full-size float expansion cannot pass merely by
        // producing the same numerical answer. Staging zero must retain every byte row.
        EXPECT_EQ(allocations.get_bytes_counter().peak, staging_bytes + conversion_bytes);
        EXPECT_EQ(allocations.get_bytes_counter().value, 0);

        allocations.push_counters();
        const auto device = this->fit_byte_device(params, weighted);
        this->expect_result_near(expected[weighted], device);
        EXPECT_EQ(allocations.get_bytes_counter().peak, conversion_bytes);
        EXPECT_EQ(allocations.get_bytes_counter().value, 0);
        allocations.pop_counters();
      }
    }
  }
}

TYPED_TEST(KmeansByteFitTest, RejectsMultiGpuHandle)
{
  raft::resources multi_gpu_handle;
  (void)raft::resource::get_multi_gpu_resource(multi_gpu_handle);

  auto d_centroids = raft::make_device_matrix<float, int64_t>(
    this->handle, TestFixture::n_clusters, TestFixture::n_features);
  float inertia  = 0.0f;
  int64_t n_iter = 0;
  auto params    = this->make_params(cuvs::cluster::kmeans::params::Array);

  EXPECT_THROW(cuvs::cluster::kmeans::fit(
                 multi_gpu_handle,
                 params,
                 raft::make_host_matrix_view<const TypeParam, int64_t>(
                   this->h_byte_input.data(), TestFixture::n_samples, TestFixture::n_features),
                 std::optional<raft::host_vector_view<const float, int64_t>>{std::nullopt},
                 d_centroids.view(),
                 raft::make_host_scalar_view<float>(&inertia),
                 raft::make_host_scalar_view<int64_t>(&n_iter)),
               raft::logic_error);
}

TEST(KmeansBatchSizingTest, UsesEightyPercentWithCap)
{
  using cuvs::cluster::kmeans::detail::kmeans_workspace_budget;
  constexpr std::size_t mib = std::size_t{1} << 20;
  EXPECT_EQ(kmeans_workspace_budget(100 * mib), 80 * mib);
  EXPECT_EQ(kmeans_workspace_budget(1024 * mib), 512 * mib);
  EXPECT_EQ(kmeans_workspace_budget(std::numeric_limits<std::size_t>::max()), 512 * mib);
}

TEST(KmeansBatchSizingTest, RoundsAndClampsConversionRows)
{
  using cuvs::cluster::kmeans::detail::kmeans_batch_rows;
  constexpr int64_t n_rows            = 1000;
  constexpr std::size_t bytes_per_row = 100;

  EXPECT_EQ(kmeans_batch_rows(n_rows, 250 * bytes_per_row, bytes_per_row), 192);
  EXPECT_EQ(kmeans_batch_rows(n_rows, 63 * bytes_per_row, bytes_per_row), 50);
  EXPECT_EQ(kmeans_batch_rows(n_rows, 79 * bytes_per_row, bytes_per_row), 63);
  EXPECT_EQ(kmeans_batch_rows(n_rows, 80 * bytes_per_row, bytes_per_row), 64);
  EXPECT_EQ(kmeans_batch_rows(int64_t{50}, 1000 * bytes_per_row, bytes_per_row), 50);
  EXPECT_EQ(kmeans_batch_rows(n_rows, 1, std::size_t{0}), n_rows);
  EXPECT_EQ(kmeans_batch_rows(int64_t{0}, 1, bytes_per_row), 0);
}

}  // namespace

class KmeansBatchLoaderTest : public ::testing::TestWithParam<bool> {};

TEST_P(KmeansBatchLoaderTest, CyclicFourPasses)
{
  constexpr int64_t n_rows     = 257;
  constexpr int64_t n_cols     = 17;
  constexpr int64_t batch_size = 64;
  constexpr int n_passes       = 4;

  raft::resources handle;
  rmm::cuda_stream copy_stream(rmm::cuda_stream::flags::non_blocking);
  std::vector<int64_t> host_data(n_rows * n_cols);
  for (int64_t row = 0; row < n_rows; ++row) {
    for (int64_t col = 0; col < n_cols; ++col) {
      host_data[row * n_cols + col] = row * n_cols + col;
    }
  }

  auto host_view =
    raft::make_host_matrix_view<const int64_t, int64_t>(host_data.data(), n_rows, n_cols);
  const bool enable_prefetch = GetParam();
  cluster::kmeans::detail::kmeans_batch_loader<int64_t, int64_t, false> loader(
    handle,
    host_view,
    batch_size,
    copy_stream,
    raft::resource::get_workspace_resource_ref(handle),
    enable_prefetch);
  auto device_readback =
    raft::make_device_vector<int64_t, int64_t>(handle, n_passes * n_rows * n_cols);

  loader.start();
  // Starting an active pipeline is a no-op.
  loader.start();
  for (int pass = 0; pass < n_passes; ++pass) {
    for (std::size_t pos = 0; pos < loader.num_batches(); ++pos) {
      const auto batch = loader.acquire(pos);
      const auto output_offset =
        (static_cast<std::size_t>(pass) * n_rows + batch.offset()) * n_cols;
      raft::copy(device_readback.data_handle() + output_offset,
                 batch.data(),
                 batch.size() * n_cols,
                 raft::resource::get_cuda_stream(handle));

      if (enable_prefetch && (pos + 1 < loader.num_batches() || pass + 1 < n_passes)) {
        loader.prefetch((pos + 1) % loader.num_batches());
      }
      const std::size_t recycle_offset = enable_prefetch ? 2 : 1;
      const bool needs_future_batch =
        pos + recycle_offset < loader.num_batches() || pass + 1 < n_passes;
      if (needs_future_batch) {
        loader.recycle(batch, (pos + recycle_offset) % loader.num_batches());
      } else {
        loader.release(batch);
      }
    }
  }

  std::vector<int64_t> readback(device_readback.size());
  raft::copy(readback.data(),
             device_readback.data_handle(),
             device_readback.size(),
             raft::resource::get_cuda_stream(handle));
  raft::resource::sync_stream(handle);
  for (int pass = 0; pass < n_passes; ++pass) {
    for (std::size_t i = 0; i < host_data.size(); ++i) {
      EXPECT_EQ(readback[static_cast<std::size_t>(pass) * host_data.size() + i], host_data[i]);
    }
  }
}

INSTANTIATE_TEST_CASE_P(WithAndWithoutPrefetch,
                        KmeansBatchLoaderTest,
                        ::testing::Values(false, true));

}  // namespace cuvs
