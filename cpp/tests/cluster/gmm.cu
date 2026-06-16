/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <cuvs/cluster/gmm.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cuvs::cluster::gmm {

template <typename T>
struct GMMInputs {
  int n_row;
  int n_col;
  int n_components;
  covariance_type cov_type;
};

// Number of elements in a covariance-typed buffer for the given type.
inline int64_t cov_len(covariance_type ct, int d, int K)
{
  switch (ct) {
    case covariance_type::FULL: return (int64_t)K * d * d;
    case covariance_type::TIED: return (int64_t)d * d;
    case covariance_type::DIAG: return (int64_t)K * d;
    case covariance_type::SPHERICAL: return (int64_t)K;
  }
  return 0;
}

template <typename T>
class GMMTest : public ::testing::TestWithParam<GMMInputs<T>> {
 protected:
  GMMTest() : stream(raft::resource::get_cuda_stream(handle)) {}

  void basicTest()
  {
    auto p = ::testing::TestWithParam<GMMInputs<T>>::GetParam();
    int n = p.n_row, d = p.n_col, K = p.n_components;
    int64_t cn = cov_len(p.cov_type, d, K);

    // Well-separated blobs: hard labels should recover the generating clusters.
    auto d_X    = raft::make_device_matrix<T, int64_t>(handle, n, d);
    auto d_yref = raft::make_device_vector<int, int>(handle, n);
    raft::random::make_blobs<T, int>(d_X.data_handle(),
                                     d_yref.data_handle(),
                                     n,
                                     d,
                                     K,
                                     stream,
                                     /* row_major          */ true,
                                     /* centers            */ nullptr,
                                     /* cluster_std        */ nullptr,
                                     /* cluster_std_scalar */ T(1.0),
                                     /* shuffle            */ false,
                                     /* center_box_min     */ static_cast<T>(-10.0f),
                                     /* center_box_max     */ static_cast<T>(10.0f),
                                     /* seed               */ 1234ULL);

    auto weights = raft::make_device_vector<T, int64_t>(handle, K);
    auto means   = raft::make_device_matrix<T, int64_t>(handle, K, d);
    auto covs    = raft::make_device_vector<T, int64_t>(handle, cn);
    auto pchol   = raft::make_device_vector<T, int64_t>(handle, cn);
    auto precs   = raft::make_device_vector<T, int64_t>(handle, cn);
    auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

    params prm;
    prm.n_components = K;
    prm.cov_type     = p.cov_type;
    prm.init         = init_method::KMeans;
    prm.max_iter     = 100;
    prm.seed         = 1234ULL;

    T lower_bound  = 0;
    int n_iter     = 0;
    bool converged = false;

    fit(handle,
        prm,
        raft::make_const_mdspan(d_X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lower_bound),
        raft::make_host_scalar_view(&n_iter),
        raft::make_host_scalar_view(&converged));

    // Fit diagnostics are sane. Well-separated blobs converge well before
    // max_iter, so the converged flag must be set and EM must have stopped early.
    ASSERT_TRUE(std::isfinite((double)lower_bound));
    ASSERT_GE(n_iter, 1);
    ASSERT_TRUE(converged) << "EM did not converge on well-separated blobs";
    ASSERT_LT(n_iter, prm.max_iter) << "converged flag set but EM ran to max_iter";

    // Hard labels recover the ground-truth clusters on well-separated blobs.
    double ari_fit =
      raft::stats::adjusted_rand_index(d_yref.data_handle(), labels.data_handle(), n, stream);
    raft::resource::sync_stream(handle, stream);
    ASSERT_GT(ari_fit, 0.95) << "fit labels disagree with ground truth";

    // predict() on the same data reproduces the fit labels exactly.
    auto labels2 = raft::make_device_vector<int, int64_t>(handle, n);
    predict(handle,
            prm,
            raft::make_const_mdspan(d_X.view()),
            raft::make_const_mdspan(weights.view()),
            raft::make_const_mdspan(means.view()),
            raft::make_const_mdspan(pchol.view()),
            labels2.view());
    double ari_pred =
      raft::stats::adjusted_rand_index(labels.data_handle(), labels2.data_handle(), n, stream);
    raft::resource::sync_stream(handle, stream);
    ASSERT_NEAR(ari_pred, 1.0, 1e-6) << "predict disagrees with fit labels";

    // predict_proba rows form a valid distribution (sum to 1, non-negative).
    auto resp = raft::make_device_matrix<T, int64_t>(handle, n, K);
    predict_proba(handle,
                  prm,
                  raft::make_const_mdspan(d_X.view()),
                  raft::make_const_mdspan(weights.view()),
                  raft::make_const_mdspan(means.view()),
                  raft::make_const_mdspan(pchol.view()),
                  resp.view());
    std::vector<T> h_resp((size_t)n * K);
    raft::update_host(h_resp.data(), resp.data_handle(), (size_t)n * K, stream);

    // score_samples: per-sample log-likelihood; its mean equals the lower
    // bound returned by fit (both are the average log p(x)).
    auto logp = raft::make_device_vector<T, int64_t>(handle, n);
    score_samples(handle,
                  prm,
                  raft::make_const_mdspan(d_X.view()),
                  raft::make_const_mdspan(weights.view()),
                  raft::make_const_mdspan(means.view()),
                  raft::make_const_mdspan(pchol.view()),
                  logp.view());
    std::vector<T> h_logp(n);
    raft::update_host(h_logp.data(), logp.data_handle(), n, stream);
    raft::resource::sync_stream(handle, stream);

    for (int i = 0; i < n; ++i) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) {
        T r = h_resp[(size_t)i * K + k];
        ASSERT_GE((double)r, -1e-5);
        s += (double)r;
      }
      ASSERT_NEAR(s, 1.0, 1e-3) << "responsibilities row " << i << " not normalized";
    }

    double mean_logp = 0.0;
    for (int i = 0; i < n; ++i)
      mean_logp += (double)h_logp[i];
    mean_logp /= n;
    // tolerance loose for float; lower_bound is the fit-time average log p(x).
    double tol = std::is_same_v<T, float> ? 1e-2 : 1e-5;
    ASSERT_NEAR(mean_logp, (double)lower_bound, std::abs((double)lower_bound) * tol + tol);
  }

  raft::resources handle;
  cudaStream_t stream;
};

const std::vector<GMMInputs<float>> inputsf = {
  {600, 8, 4, covariance_type::FULL},
  {600, 8, 4, covariance_type::TIED},
  {600, 8, 4, covariance_type::DIAG},
  {600, 8, 4, covariance_type::SPHERICAL},
  {2000, 16, 5, covariance_type::FULL},  // fixed-D=16 specialization
  {2000, 16, 5, covariance_type::DIAG},
  {2000, 32, 4, covariance_type::FULL},          // fixed-D=32 specialization
  {2000, 50, 4, covariance_type::FULL},          // fixed-D=50 specialization
  {2000, 64, 4, covariance_type::FULL},          // fixed-D=64 specialization (boundary)
  {3000, 128, 4, covariance_type::FULL},         // 64<d<257 -> tiled thread64 kernel
  {3000, 128, 4, covariance_type::TIED},         // tied, tiled thread64 kernel
  {4000, 300, 4, covariance_type::FULL},         // d>=257 (float) -> cuBLAS E-step route
  {2000, 512, 16, covariance_type::DIAG},        // K*d large -> diag global-mem path
  {2000, 1024, 16, covariance_type::SPHERICAL},  // K*d large -> spherical global-mem path
};

const std::vector<GMMInputs<double>> inputsd = {
  {600, 8, 4, covariance_type::FULL},
  {600, 8, 4, covariance_type::TIED},
  {600, 8, 4, covariance_type::DIAG},
  {600, 8, 4, covariance_type::SPHERICAL},
  {2000, 16, 5, covariance_type::FULL},
  {2000, 50, 4, covariance_type::FULL},   // fixed-D=50 specialization
  {2000, 64, 4, covariance_type::FULL},   // fixed-D=64 specialization (boundary)
  {3000, 128, 4, covariance_type::FULL},  // d>64 (double) -> cuBLAS E-step route
};

using GMMTestF = GMMTest<float>;
TEST_P(GMMTestF, Result) { basicTest(); }
INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestF, ::testing::ValuesIn(inputsf));

using GMMTestD = GMMTest<double>;
TEST_P(GMMTestD, Result) { basicTest(); }
INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestD, ::testing::ValuesIn(inputsd));

// ---------------------------------------------------------------------------
// Standalone tests for behaviors not covered by the parametrized sweep:
// every init method, warm_start, n_init best-of, and the ill-defined-
// covariance error path.
// ---------------------------------------------------------------------------
namespace {

// Generate well-separated blobs into freshly allocated device buffers.
template <typename T>
std::pair<raft::device_matrix<T, int64_t>, raft::device_vector<int, int>> make_gmm_blobs(
  raft::resources const& handle, int n, int d, int K, std::uint64_t seed = 1234ULL)
{
  auto X    = raft::make_device_matrix<T, int64_t>(handle, n, d);
  auto yref = raft::make_device_vector<int, int>(handle, n);
  raft::random::make_blobs<T, int>(X.data_handle(),
                                   yref.data_handle(),
                                   n,
                                   d,
                                   K,
                                   raft::resource::get_cuda_stream(handle),
                                   true,
                                   nullptr,
                                   nullptr,
                                   T(1.0),
                                   false,
                                   static_cast<T>(-10.0f),
                                   static_cast<T>(10.0f),
                                   seed);
  return {std::move(X), std::move(yref)};
}

}  // namespace

// Every init method produces a valid fit; kmeans-family inits recover the
// generating clusters on well-separated blobs.
TEST(GMMExtra, InitMethods)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const int n = 1500, d = 8, K = 4;
  auto [X, yref] = make_gmm_blobs<float>(handle, n, d, K);

  for (auto im : {init_method::KMeans,
                  init_method::KMeansPlusPlus,
                  init_method::Random,
                  init_method::RandomFromData}) {
    int64_t cn   = cov_len(covariance_type::FULL, d, K);
    auto weights = raft::make_device_vector<float, int64_t>(handle, K);
    auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
    auto covs    = raft::make_device_vector<float, int64_t>(handle, cn);
    auto pchol   = raft::make_device_vector<float, int64_t>(handle, cn);
    auto precs   = raft::make_device_vector<float, int64_t>(handle, cn);
    auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

    params prm;
    prm.n_components = K;
    prm.cov_type     = covariance_type::FULL;
    prm.init         = im;
    prm.n_init       = 1;
    // A modest regularizer keeps every init's first covariance well-defined so
    // the test deterministically exercises the init code path (random inits
    // can otherwise collapse a component, which legitimately raises).
    prm.reg_covar = 1e-2;
    prm.max_iter  = 100;
    prm.seed      = 1234ULL;

    float lb = 0;
    int it   = 0;
    bool cv  = false;
    fit(handle,
        prm,
        raft::make_const_mdspan(X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lb),
        raft::make_host_scalar_view(&it),
        raft::make_host_scalar_view(&cv));

    ASSERT_TRUE(std::isfinite(lb)) << "init " << (int)im;
    if (im == init_method::KMeans || im == init_method::KMeansPlusPlus) {
      double ari =
        raft::stats::adjusted_rand_index(yref.data_handle(), labels.data_handle(), n, stream);
      raft::resource::sync_stream(handle, stream);
      ASSERT_GT(ari, 0.95) << "init " << (int)im;
    }
  }
}

// warm_start reuses the supplied weights/means/covariances as the single
// initialization and refines them to a finite, non-decreasing lower bound.
TEST(GMMExtra, WarmStart)
{
  raft::resources handle;
  const int n = 1500, d = 8, K = 4;
  auto [X, yref] = make_gmm_blobs<double>(handle, n, d, K);

  int64_t cn   = cov_len(covariance_type::FULL, d, K);
  auto weights = raft::make_device_vector<double, int64_t>(handle, K);
  auto means   = raft::make_device_matrix<double, int64_t>(handle, K, d);
  auto covs    = raft::make_device_vector<double, int64_t>(handle, cn);
  auto pchol   = raft::make_device_vector<double, int64_t>(handle, cn);
  auto precs   = raft::make_device_vector<double, int64_t>(handle, cn);
  auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

  params prm;
  prm.n_components = K;
  prm.cov_type     = covariance_type::FULL;
  prm.init         = init_method::KMeans;
  prm.max_iter     = 5;
  prm.seed         = 1234ULL;

  double lb1 = 0;
  int it1    = 0;
  bool cv1   = false;
  auto run   = [&](bool warm, double& lb, int& it, bool& cv) {
    fit(handle,
        prm,
        raft::make_const_mdspan(X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lb),
        raft::make_host_scalar_view(&it),
        raft::make_host_scalar_view(&cv),
        warm);
  };
  run(false, lb1, it1, cv1);

  // Continue from the fitted parameters; the lower bound should not regress.
  double lb2   = 0;
  int it2      = 0;
  bool cv2     = false;
  prm.max_iter = 20;
  run(true, lb2, it2, cv2);
  ASSERT_TRUE(std::isfinite(lb2));
  ASSERT_GE(lb2, lb1 - 1e-6);
}

// n_init>1 keeps the restart with the largest lower bound. The first restart
// of an N-restart fit uses the same seed as a single-restart fit, so more
// restarts can only match or beat it: lower_bound(n_init=N) >= lower_bound(1).
TEST(GMMExtra, NInitSelectsBest)
{
  raft::resources handle;
  const int n = 1500, d = 8, K = 4;
  auto [X, yref] = make_gmm_blobs<float>(handle, n, d, K);

  int64_t cn   = cov_len(covariance_type::FULL, d, K);
  auto weights = raft::make_device_vector<float, int64_t>(handle, K);
  auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
  auto covs    = raft::make_device_vector<float, int64_t>(handle, cn);
  auto pchol   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto precs   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

  auto run = [&](int n_init) {
    params prm;
    prm.n_components = K;
    prm.cov_type     = covariance_type::FULL;
    prm.init         = init_method::Random;  // restart-sensitive init
    prm.n_init       = n_init;
    prm.reg_covar    = 1e-2;  // keep every restart well-defined
    prm.max_iter     = 50;
    prm.seed         = 1234ULL;
    float lb         = 0;
    int it           = 0;
    bool cv          = false;
    fit(handle,
        prm,
        raft::make_const_mdspan(X.view()),
        weights.view(),
        means.view(),
        covs.view(),
        pchol.view(),
        precs.view(),
        labels.view(),
        raft::make_host_scalar_view(&lb),
        raft::make_host_scalar_view(&it),
        raft::make_host_scalar_view(&cv));
    return lb;
  };

  float lb1  = run(1);
  float lb10 = run(10);
  ASSERT_GE((double)lb10, (double)lb1 - 1e-5);
}

// A degenerate component (more components than distinct points) yields an
// ill-defined covariance and must surface as an exception rather than NaNs.
TEST(GMMExtra, IllDefinedCovarianceThrows)
{
  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  const int n = 6, d = 4, K = 5;

  // All points identical -> any component covariance collapses to zero.
  auto X = raft::make_device_matrix<float, int64_t>(handle, n, d);
  RAFT_CUDA_TRY(cudaMemsetAsync(X.data_handle(), 0, sizeof(float) * (size_t)n * d, stream));

  int64_t cn   = cov_len(covariance_type::FULL, d, K);
  auto weights = raft::make_device_vector<float, int64_t>(handle, K);
  auto means   = raft::make_device_matrix<float, int64_t>(handle, K, d);
  auto covs    = raft::make_device_vector<float, int64_t>(handle, cn);
  auto pchol   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto precs   = raft::make_device_vector<float, int64_t>(handle, cn);
  auto labels  = raft::make_device_vector<int, int64_t>(handle, n);

  params prm;
  prm.n_components = K;
  prm.cov_type     = covariance_type::FULL;
  prm.init         = init_method::RandomFromData;
  prm.reg_covar    = 0.0;  // disable the regularizer that would otherwise mask it
  prm.max_iter     = 50;
  prm.seed         = 1234ULL;

  float lb = 0;
  int it   = 0;
  bool cv  = false;
  EXPECT_ANY_THROW(fit(handle,
                       prm,
                       raft::make_const_mdspan(X.view()),
                       weights.view(),
                       means.view(),
                       covs.view(),
                       pchol.view(),
                       precs.view(),
                       labels.view(),
                       raft::make_host_scalar_view(&lb),
                       raft::make_host_scalar_view(&it),
                       raft::make_host_scalar_view(&cv)));
}

}  // namespace cuvs::cluster::gmm
