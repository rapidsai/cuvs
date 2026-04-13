/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/cluster/kmeans.hpp>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace cuvs {

template <typename T>
struct KmeansSNMGInputs {
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  int weight_mode;  // 0 = no weights, 1 = uniform (all-ones), 2 = non-uniform
  int streaming_batch_size;
  int n_init;
  cuvs::cluster::kmeans::params::InitMethod init = cuvs::cluster::kmeans::params::Array;
  bool inertia_check                             = true;
  int max_iter                                   = 20;
};

template <typename T>
class KmeansSNMGTest : public ::testing::TestWithParam<KmeansSNMGInputs<T>> {
 protected:
  KmeansSNMGTest() : clique_() { clique_.set_memory_pool(50); }

  void runTest()
  {
    testparams_ = ::testing::TestWithParam<KmeansSNMGInputs<T>>::GetParam();

    int n_samples  = testparams_.n_row;
    int n_features = testparams_.n_col;
    int n_clusters = testparams_.n_clusters;
    int num_ranks  = raft::resource::get_num_ranks(clique_);

    auto stream = raft::resource::get_cuda_stream(clique_);

    auto X      = raft::make_device_matrix<T, int>(clique_, n_samples, n_features);
    auto labels = raft::make_device_vector<int, int>(clique_, n_samples);

    raft::random::make_blobs<T, int>(X.data_handle(),
                                     labels.data_handle(),
                                     n_samples,
                                     n_features,
                                     n_clusters,
                                     stream,
                                     true,
                                     nullptr,
                                     nullptr,
                                     T(1.0),
                                     false,
                                     (T)-10.0f,
                                     (T)10.0f,
                                     (uint64_t)1234);

    // Copy X to host
    std::vector<T> h_X(n_samples * n_features);
    raft::update_host(h_X.data(), X.data_handle(), n_samples * n_features, stream);
    raft::resource::sync_stream(clique_, stream);

    auto h_X_view =
      raft::make_host_matrix_view<const T, int64_t>(h_X.data(), n_samples, n_features);

    auto d_centroids_snmg = raft::make_device_matrix<T, int64_t>(clique_, n_clusters, n_features);
    auto d_centroids_ref  = raft::make_device_matrix<T, int64_t>(clique_, n_clusters, n_features);

    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      raft::random::RngState rng(42);
      raft::random::uniform(
        clique_, rng, d_centroids_snmg.data_handle(), n_clusters * n_features, T(-1), T(1));
      raft::copy(d_centroids_ref.data_handle(),
                 d_centroids_snmg.data_handle(),
                 n_clusters * n_features,
                 stream);
      raft::resource::sync_stream(clique_, stream);
    }

    // --- Prepare sample weights ---
    std::optional<raft::host_vector_view<const T, int64_t>> h_sw = std::nullopt;
    std::vector<T> h_sample_weight;
    if (testparams_.weight_mode > 0) {
      h_sample_weight.resize(n_samples);
      for (int i = 0; i < n_samples; ++i) {
        h_sample_weight[i] = (testparams_.weight_mode == 2) ? T(1) + T(i % 5) : T(1);
      }
      h_sw = raft::make_host_vector_view<const T, int64_t>(h_sample_weight.data(), n_samples);
    }

    // --- Run SNMG fit ---
    cuvs::cluster::kmeans::params snmg_params;
    snmg_params.n_clusters           = n_clusters;
    snmg_params.tol                  = testparams_.tol;
    snmg_params.max_iter             = testparams_.max_iter;
    snmg_params.n_init               = testparams_.n_init;
    snmg_params.rng_state.seed       = 42;
    snmg_params.init                 = testparams_.init;
    snmg_params.inertia_check        = testparams_.inertia_check;
    snmg_params.streaming_batch_size = testparams_.streaming_batch_size;

    T snmg_inertia      = T{0};
    int64_t snmg_n_iter = 0;

    cuvs::cluster::kmeans::fit(clique_,
                               snmg_params,
                               h_X_view,
                               h_sw,
                               d_centroids_snmg.view(),
                               raft::make_host_scalar_view(&snmg_inertia),
                               raft::make_host_scalar_view(&snmg_n_iter));

    raft::resource::sync_stream(clique_, stream);

    // --- Run single-GPU reference fit ---
    raft::resources sg_handle;
    auto sg_stream = raft::resource::get_cuda_stream(sg_handle);

    auto d_centroids_sg = raft::make_device_matrix<T, int64_t>(sg_handle, n_clusters, n_features);
    if (testparams_.init == cuvs::cluster::kmeans::params::Array) {
      raft::copy(d_centroids_sg.data_handle(),
                 d_centroids_ref.data_handle(),
                 n_clusters * n_features,
                 sg_stream);
      raft::resource::sync_stream(sg_handle, sg_stream);
    }

    cuvs::cluster::kmeans::params sg_params = snmg_params;

    T sg_inertia      = T{0};
    int64_t sg_n_iter = 0;

    cuvs::cluster::kmeans::fit(sg_handle,
                               sg_params,
                               h_X_view,
                               h_sw,
                               d_centroids_sg.view(),
                               raft::make_host_scalar_view(&sg_inertia),
                               raft::make_host_scalar_view(&sg_n_iter));

    raft::resource::sync_stream(sg_handle, sg_stream);

    // --- Predict labels using both centroid sets on single GPU ---
    rmm::device_uvector<int> d_labels_snmg(n_samples, sg_stream);
    rmm::device_uvector<int> d_labels_sg(n_samples, sg_stream);
    rmm::device_uvector<int> d_labels_ref(n_samples, sg_stream);

    raft::copy(d_labels_ref.data(), labels.data_handle(), n_samples, sg_stream);

    auto X_dev_view =
      raft::make_device_matrix_view<const T, int>(X.data_handle(), n_samples, n_features);

    cuvs::cluster::kmeans::params pred_params;
    pred_params.n_clusters = n_clusters;

    // Copy SNMG centroids to single-GPU handle for predict
    auto d_centroids_snmg_copy =
      raft::make_device_matrix<T, int>(sg_handle, n_clusters, n_features);
    raft::copy(d_centroids_snmg_copy.data_handle(),
               d_centroids_snmg.data_handle(),
               n_clusters * n_features,
               sg_stream);

    auto d_centroids_sg_int = raft::make_device_matrix<T, int>(sg_handle, n_clusters, n_features);
    raft::copy(d_centroids_sg_int.data_handle(),
               d_centroids_sg.data_handle(),
               n_clusters * n_features,
               sg_stream);

    T pred_inertia_snmg = T{0};
    cuvs::cluster::kmeans::predict(
      sg_handle,
      pred_params,
      X_dev_view,
      std::nullopt,
      raft::make_device_matrix_view<const T, int>(
        d_centroids_snmg_copy.data_handle(), n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_snmg.data(), n_samples),
      true,
      raft::make_host_scalar_view(&pred_inertia_snmg));

    T pred_inertia_sg = T{0};
    cuvs::cluster::kmeans::predict(
      sg_handle,
      pred_params,
      X_dev_view,
      std::nullopt,
      raft::make_device_matrix_view<const T, int>(
        d_centroids_sg_int.data_handle(), n_clusters, n_features),
      raft::make_device_vector_view<int, int>(d_labels_sg.data(), n_samples),
      true,
      raft::make_host_scalar_view(&pred_inertia_sg));

    raft::resource::sync_stream(sg_handle, sg_stream);

    // --- Evaluate: compare SNMG labels with reference (make_blobs) labels ---
    ari_vs_ref_ = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels_snmg.data(), n_samples, sg_stream);

    // ARI between SNMG and single-GPU results
    ari_vs_sg_ = raft::stats::adjusted_rand_index(
      d_labels_sg.data(), d_labels_snmg.data(), n_samples, sg_stream);

    raft::resource::sync_stream(sg_handle, sg_stream);

    snmg_inertia_ = snmg_inertia;
    sg_inertia_   = sg_inertia;
    snmg_n_iter_  = snmg_n_iter;
    sg_n_iter_    = sg_n_iter;

    if (ari_vs_ref_ < 0.94 || ari_vs_sg_ < 0.94) {
      std::cout << "SNMG KMeans: ARI vs ref = " << ari_vs_ref_ << ", ARI vs SG = " << ari_vs_sg_
                << ", num_ranks = " << num_ranks << ", snmg_inertia = " << snmg_inertia
                << ", sg_inertia = " << sg_inertia << ", snmg_n_iter = " << snmg_n_iter
                << ", sg_n_iter = " << sg_n_iter << std::endl;
    }
  }

  void SetUp() override { runTest(); }

  raft::device_resources_snmg clique_;
  KmeansSNMGInputs<T> testparams_;
  double ari_vs_ref_   = 0;
  double ari_vs_sg_    = 0;
  T snmg_inertia_      = T{0};
  T sg_inertia_        = T{0};
  int64_t snmg_n_iter_ = 0;
  int64_t sg_n_iter_   = 0;
};

// ============================================================================
// Float test inputs
// ============================================================================
const std::vector<KmeansSNMGInputs<float>> snmg_inputsf = {
  // n_row, n_col, n_clusters, tol, weight_mode, streaming_batch_size, n_init[, init]
  {1000, 32, 5, 0.0001f, 0, 1000, 1},
  {1000, 32, 5, 0.0001f, 1, 1000, 1},
  {1000, 32, 5, 0.0001f, 0, 128, 1},
  {10000, 16, 10, 0.0001f, 0, 2000, 1},
  {10000, 16, 10, 0.0001f, 1, 2000, 1},
  {10000, 16, 10, 0.0001f, 0, 500, 1},
  {1001, 32, 5, 0.0001f, 0, 1001, 1},
  {1000, 32, 5, 0.0001f, 0, 1000, 3},
  {1000, 32, 5, 0.0001f, 0, 1000, 1, cuvs::cluster::kmeans::params::KMeansPlusPlus},
  {1001, 32, 5, 0.0001f, 0, 128, 1},
  // Non-uniform weights: exercises weight_scale = global_n / global_wt normalization
  {1000, 32, 5, 0.0001f, 2, 1000, 1},
  {10000, 16, 10, 0.0001f, 2, 2000, 1},
  // Extreme batch size = 1: single-element work buffers, many batch iterations
  {100, 8, 3, 0.001f, 0, 1, 1},
  // Very small dataset: some ranks may get only 2-3 rows with 4+ GPUs
  {10, 4, 3, 0.001f, 0, 10, 1},
  // Trivial single cluster: convergence should be immediate
  {1000, 16, 1, 0.0001f, 0, 1000, 1},
  // Batch size > n_samples: tests per-rank clamping logic
  {1000, 32, 5, 0.0001f, 0, 5000, 1},
  // n_init > 1 with KMeansPlusPlus: best-of-n seed management across ranks
  {1000, 32, 5, 0.0001f, 0, 1000, 3, cuvs::cluster::kmeans::params::KMeansPlusPlus},
  // inertia_check=false: convergence only via centroid shift
  {1000, 32, 5, 0.0001f, 0, 1000, 1, cuvs::cluster::kmeans::params::Array, false},
  {1000, 32, 5, 0.0001f, 0, 128, 1, cuvs::cluster::kmeans::params::Array, false},
  // max_iter saturation: algorithm should stop at max_iter without convergence
  {1000, 32, 5, 0.0001f, 0, 1000, 1, cuvs::cluster::kmeans::params::Array, true, 2},
};

// ============================================================================
// Double test inputs
// ============================================================================
const std::vector<KmeansSNMGInputs<double>> snmg_inputsd = {
  {1000, 32, 5, 0.0001, 0, 1000, 1},
  {1000, 32, 5, 0.0001, 0, 128, 1},
  {1000, 32, 5, 0.0001, 1, 1000, 1},
  // Non-uniform weights for double precision
  {1000, 32, 5, 0.0001, 2, 1000, 1},
};

// ============================================================================
// Test fixtures
// ============================================================================
typedef KmeansSNMGTest<float> KmeansSNMGTestF;
typedef KmeansSNMGTest<double> KmeansSNMGTestD;

TEST_P(KmeansSNMGTestF, Result)
{
  ASSERT_GE(ari_vs_ref_, 0.94);
  ASSERT_GE(ari_vs_sg_, 0.94);
  ASSERT_GT(snmg_n_iter_, int64_t{0});
  ASSERT_LE(snmg_n_iter_, static_cast<int64_t>(testparams_.max_iter));
  if (testparams_.init == cuvs::cluster::kmeans::params::Array && sg_inertia_ > 0) {
    EXPECT_LT(std::abs(snmg_inertia_ - sg_inertia_) / sg_inertia_, decltype(sg_inertia_){0.05});
  }
}

TEST_P(KmeansSNMGTestD, Result)
{
  ASSERT_GE(ari_vs_ref_, 0.94);
  ASSERT_GE(ari_vs_sg_, 0.94);
  ASSERT_GT(snmg_n_iter_, int64_t{0});
  ASSERT_LE(snmg_n_iter_, static_cast<int64_t>(testparams_.max_iter));
  if (testparams_.init == cuvs::cluster::kmeans::params::Array && sg_inertia_ > 0) {
    EXPECT_LT(std::abs(snmg_inertia_ - sg_inertia_) / sg_inertia_, decltype(sg_inertia_){0.05});
  }
}

INSTANTIATE_TEST_CASE_P(KmeansSNMGTests, KmeansSNMGTestF, ::testing::ValuesIn(snmg_inputsf));
INSTANTIATE_TEST_CASE_P(KmeansSNMGTests, KmeansSNMGTestD, ::testing::ValuesIn(snmg_inputsd));

}  // namespace cuvs
