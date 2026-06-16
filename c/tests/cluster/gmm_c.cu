/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "test_utils.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_uvector.hpp>

#include "../../src/core/interop.hpp"
#include <cuvs/cluster/gmm.h>
#include <cuvs/core/c_api.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int64_t kNSamples  = 8;
constexpr int64_t kNFeatures = 2;
constexpr int kNComponents   = 2;

// Two tight, well-separated clusters of four points each.
float kDataset[kNSamples][kNFeatures] = {
  {1.0f, 1.0f},
  {1.0f, 2.0f},
  {2.0f, 1.0f},
  {2.0f, 2.0f},
  {10.0f, 10.0f},
  {10.0f, 11.0f},
  {11.0f, 10.0f},
  {11.0f, 11.0f},
};

void test_fit_predict()
{
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t cn = (int64_t)kNComponents * kNFeatures * kNFeatures;  // FULL

  rmm::device_uvector<float> dataset_d(kNSamples * kNFeatures, stream);
  rmm::device_uvector<float> weights_d(kNComponents, stream);
  rmm::device_uvector<float> means_d(kNComponents * kNFeatures, stream);
  rmm::device_uvector<float> covs_d(cn, stream);
  rmm::device_uvector<float> pchol_d(cn, stream);
  rmm::device_uvector<float> precs_d(cn, stream);
  rmm::device_uvector<int32_t> labels_d(kNSamples, stream);
  rmm::device_uvector<int32_t> labels2_d(kNSamples, stream);
  rmm::device_uvector<float> resp_d(kNSamples * kNComponents, stream);
  rmm::device_uvector<float> logp_d(kNSamples, stream);

  raft::copy(
    dataset_d.data(), reinterpret_cast<float const*>(kDataset), kNSamples * kNFeatures, stream);

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);

  cuvsGMMParams_t params;
  ASSERT_EQ(cuvsGMMParamsCreate(&params), CUVS_SUCCESS);
  params->n_components    = kNComponents;
  params->covariance_type = CUVS_GMM_COVARIANCE_FULL;
  params->max_iter        = 100;
  params->seed            = 1234ULL;

  auto to_dl_mat = [&](float* p, int64_t r, int64_t c, DLManagedTensor* t) {
    cuvs::core::to_dlpack(raft::make_device_matrix_view<float, int64_t>(p, r, c), t);
  };
  auto to_dl_vec = [&](float* p, int64_t n, DLManagedTensor* t) {
    cuvs::core::to_dlpack(raft::make_device_vector_view<float, int64_t>(p, n), t);
  };

  DLManagedTensor X_t{}, w_t{}, m_t{}, cov_t{}, pc_t{}, pr_t{}, lab_t{}, lab2_t{}, resp_t{}, lp_t{};
  to_dl_mat(dataset_d.data(), kNSamples, kNFeatures, &X_t);
  to_dl_vec(weights_d.data(), kNComponents, &w_t);
  to_dl_mat(means_d.data(), kNComponents, kNFeatures, &m_t);
  to_dl_vec(covs_d.data(), cn, &cov_t);
  to_dl_vec(pchol_d.data(), cn, &pc_t);
  to_dl_vec(precs_d.data(), cn, &pr_t);
  cuvs::core::to_dlpack(raft::make_device_vector_view<int32_t, int64_t>(labels_d.data(), kNSamples),
                        &lab_t);
  cuvs::core::to_dlpack(
    raft::make_device_vector_view<int32_t, int64_t>(labels2_d.data(), kNSamples), &lab2_t);
  to_dl_mat(resp_d.data(), kNSamples, kNComponents, &resp_t);
  to_dl_vec(logp_d.data(), kNSamples, &lp_t);

  double lower_bound = 0.0;
  int n_iter         = -1;
  bool converged     = false;

  ASSERT_EQ(cuvsGMMFit(res,
                       params,
                       &X_t,
                       &w_t,
                       &m_t,
                       &cov_t,
                       &pc_t,
                       &pr_t,
                       &lab_t,
                       &lower_bound,
                       &n_iter,
                       &converged,
                       /* warm_start */ false),
            CUVS_SUCCESS);
  EXPECT_GT(n_iter, 0);

  ASSERT_EQ(cuvsGMMPredict(res, params, &X_t, &w_t, &m_t, &pc_t, &lab2_t), CUVS_SUCCESS);
  ASSERT_EQ(cuvsGMMPredictProba(res, params, &X_t, &w_t, &m_t, &pc_t, &resp_t), CUVS_SUCCESS);
  ASSERT_EQ(cuvsGMMScoreSamples(res, params, &X_t, &w_t, &m_t, &pc_t, &lp_t), CUVS_SUCCESS);

  std::vector<int32_t> h_labels(kNSamples), h_labels2(kNSamples);
  raft::copy(h_labels.data(), labels_d.data(), kNSamples, stream);
  raft::copy(h_labels2.data(), labels2_d.data(), kNSamples, stream);
  std::vector<float> h_resp(kNSamples * kNComponents);
  raft::copy(h_resp.data(), resp_d.data(), kNSamples * kNComponents, stream);
  std::vector<float> h_logp(kNSamples);
  raft::copy(h_logp.data(), logp_d.data(), kNSamples, stream);
  raft::resource::sync_stream(handle, stream);

  // score_samples returns a finite per-sample log-likelihood.
  for (int i = 0; i < kNSamples; ++i)
    EXPECT_TRUE(std::isfinite(h_logp[i]));

  // fit and predict agree, and both recover the two-cluster partition (the two
  // halves get the same label within each half, different across halves).
  for (int i = 0; i < kNSamples; ++i)
    EXPECT_EQ(h_labels[i], h_labels2[i]);
  EXPECT_EQ(h_labels[0], h_labels[1]);
  EXPECT_EQ(h_labels[0], h_labels[2]);
  EXPECT_EQ(h_labels[0], h_labels[3]);
  EXPECT_EQ(h_labels[4], h_labels[5]);
  EXPECT_NE(h_labels[0], h_labels[4]);

  // responsibilities normalized per row.
  for (int i = 0; i < kNSamples; ++i) {
    float s = h_resp[i * kNComponents] + h_resp[i * kNComponents + 1];
    EXPECT_NEAR(s, 1.0f, 1e-3f);
  }

  lp_t.deleter(&lp_t);
  resp_t.deleter(&resp_t);
  lab2_t.deleter(&lab2_t);
  lab_t.deleter(&lab_t);
  pr_t.deleter(&pr_t);
  pc_t.deleter(&pc_t);
  cov_t.deleter(&cov_t);
  m_t.deleter(&m_t);
  w_t.deleter(&w_t);
  X_t.deleter(&X_t);

  ASSERT_EQ(cuvsGMMParamsDestroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

// Exercises the float64 dispatch and the DIAG flat-buffer sizing (K*d) through
// the C boundary, plus cuvsGMMScoreSamples on the alternate dtype.
void test_fit_score_double_diag()
{
  raft::handle_t handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t cn = (int64_t)kNComponents * kNFeatures;  // DIAG

  std::vector<double> h_X(kNSamples * kNFeatures);
  for (int i = 0; i < kNSamples; ++i)
    for (int j = 0; j < kNFeatures; ++j)
      h_X[i * kNFeatures + j] = static_cast<double>(kDataset[i][j]);

  rmm::device_uvector<double> X_d(kNSamples * kNFeatures, stream);
  rmm::device_uvector<double> weights_d(kNComponents, stream);
  rmm::device_uvector<double> means_d(kNComponents * kNFeatures, stream);
  rmm::device_uvector<double> covs_d(cn, stream);
  rmm::device_uvector<double> pchol_d(cn, stream);
  rmm::device_uvector<double> precs_d(cn, stream);
  rmm::device_uvector<int32_t> labels_d(kNSamples, stream);
  rmm::device_uvector<double> logp_d(kNSamples, stream);
  raft::copy(X_d.data(), h_X.data(), kNSamples * kNFeatures, stream);

  cuvsResources_t res;
  ASSERT_EQ(cuvsResourcesCreate(&res), CUVS_SUCCESS);
  cuvsGMMParams_t params;
  ASSERT_EQ(cuvsGMMParamsCreate(&params), CUVS_SUCCESS);
  params->n_components    = kNComponents;
  params->covariance_type = CUVS_GMM_COVARIANCE_DIAG;
  params->max_iter        = 100;
  params->seed            = 1234ULL;

  auto dmat = [&](double* p, int64_t r, int64_t c, DLManagedTensor* t) {
    cuvs::core::to_dlpack(raft::make_device_matrix_view<double, int64_t>(p, r, c), t);
  };
  auto dvec = [&](double* p, int64_t n, DLManagedTensor* t) {
    cuvs::core::to_dlpack(raft::make_device_vector_view<double, int64_t>(p, n), t);
  };

  DLManagedTensor X_t{}, w_t{}, m_t{}, cov_t{}, pc_t{}, pr_t{}, lab_t{}, lp_t{};
  dmat(X_d.data(), kNSamples, kNFeatures, &X_t);
  dvec(weights_d.data(), kNComponents, &w_t);
  dmat(means_d.data(), kNComponents, kNFeatures, &m_t);
  dvec(covs_d.data(), cn, &cov_t);
  dvec(pchol_d.data(), cn, &pc_t);
  dvec(precs_d.data(), cn, &pr_t);
  cuvs::core::to_dlpack(raft::make_device_vector_view<int32_t, int64_t>(labels_d.data(), kNSamples),
                        &lab_t);
  dvec(logp_d.data(), kNSamples, &lp_t);

  double lower_bound = 0.0;
  int n_iter         = -1;
  bool converged     = false;
  ASSERT_EQ(cuvsGMMFit(res,
                       params,
                       &X_t,
                       &w_t,
                       &m_t,
                       &cov_t,
                       &pc_t,
                       &pr_t,
                       &lab_t,
                       &lower_bound,
                       &n_iter,
                       &converged,
                       /* warm_start */ false),
            CUVS_SUCCESS);
  EXPECT_GT(n_iter, 0);
  ASSERT_EQ(cuvsGMMScoreSamples(res, params, &X_t, &w_t, &m_t, &pc_t, &lp_t), CUVS_SUCCESS);

  std::vector<double> h_logp(kNSamples);
  raft::copy(h_logp.data(), logp_d.data(), kNSamples, stream);
  raft::resource::sync_stream(handle, stream);
  for (int i = 0; i < kNSamples; ++i)
    EXPECT_TRUE(std::isfinite(h_logp[i]));

  lp_t.deleter(&lp_t);
  lab_t.deleter(&lab_t);
  pr_t.deleter(&pr_t);
  pc_t.deleter(&pc_t);
  cov_t.deleter(&cov_t);
  m_t.deleter(&m_t);
  w_t.deleter(&w_t);
  X_t.deleter(&X_t);
  ASSERT_EQ(cuvsGMMParamsDestroy(params), CUVS_SUCCESS);
  ASSERT_EQ(cuvsResourcesDestroy(res), CUVS_SUCCESS);
}

}  // namespace

TEST(GMMC, FitPredict) { test_fit_predict(); }

TEST(GMMC, FitScoreDoubleDiag) { test_fit_score_double_diag(); }

TEST(GMMC, ParamsCreateDestroy)
{
  cuvsGMMParams_t params = nullptr;
  ASSERT_EQ(cuvsGMMParamsCreate(&params), CUVS_SUCCESS);
  ASSERT_NE(params, nullptr);
  EXPECT_GT(params->n_components, 0);
  EXPECT_GT(params->max_iter, 0);
  ASSERT_EQ(cuvsGMMParamsDestroy(params), CUVS_SUCCESS);
}
