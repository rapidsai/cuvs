/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

 #include "test_utils.cuh"

 #include <cuda_runtime.h>
 #include <gtest/gtest.h>

 #include <raft/core/handle.hpp>
 #include <raft/core/resources.hpp>
 #include <rmm/device_uvector.hpp>

 #include <cuvs/cluster/kmeans.h>
 #include <cuvs/core/c_api.h>

 #include <cstdint>
 #include <vector>

 extern "C" void run_kmeans(int64_t n_samples,
                               int64_t n_features,
                               int n_clusters,
                               int max_iter,
                               double tol,
                               cuvsKMeansInitMethod init,
                               bool dataset_on_host,
                               int64_t streaming_batch_size,
                               void* dataset_data,
                               float* centroids_data,
                               int32_t* labels_data,
                               double* inertia_out,
                               int* n_iter_out,
                               double* predict_inertia_out,
                               double* cluster_cost_out);

 // TODO(cuVS 26.08): remove run_kmeans_v2 declaration once the `_v2` ABI is
// promoted to the unsuffixed names.
extern "C" void run_kmeans_v2(int64_t n_samples,
                               int64_t n_features,
                               int n_clusters,
                               int max_iter,
                               double tol,
                               cuvsKMeansInitMethod init,
                               bool dataset_on_host,
                               int64_t streaming_batch_size,
                               void* dataset_data,
                               float* centroids_data,
                               int32_t* labels_data,
                               double* inertia_out,
                               int* n_iter_out,
                               double* predict_inertia_out,
                               double* cluster_cost_out);

 namespace {

 constexpr int64_t kNSamples  = 8;
 constexpr int64_t kNFeatures = 2;
 constexpr int kNClusters     = 2;

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

float kInitCentroids[kNClusters][kNFeatures] = {
  {0.0f, 0.0f},
  {12.0f, 12.0f},
};

 float kExpectedCentroids[kNClusters * kNFeatures] = {1.5f, 1.5f, 10.5f, 10.5f};
 int32_t kExpectedLabels[kNSamples]                = {0, 0, 0, 0, 1, 1, 1, 1};

 // 8 points, each at squared distance 0.5 from its cluster mean -> 4.0.
 constexpr double kExpectedInertia = 4.0;

 template <typename RunFn>
 void test_fit_predict(RunFn run_fn)
 {
   raft::handle_t handle;
   auto stream = raft::resource::get_cuda_stream(handle);

   rmm::device_uvector<float> dataset_d(kNSamples * kNFeatures, stream);
   rmm::device_uvector<float> centroids_d(kNClusters * kNFeatures, stream);
   rmm::device_uvector<int32_t> labels_d(kNSamples, stream);

   raft::copy(dataset_d.data(),
              reinterpret_cast<float const*>(kDataset),
              kNSamples * kNFeatures,
              stream);
   raft::copy(centroids_d.data(),
              reinterpret_cast<float const*>(kInitCentroids),
              kNClusters * kNFeatures,
              stream);

   double inertia         = -1.0;
   int n_iter             = -1;
   double predict_inertia = -1.0;
   double cluster_cost    = -1.0;

   run_fn(kNSamples,
          kNFeatures,
          kNClusters,
          100,
          1e-6,
          Array,
          false,
          0,
          dataset_d.data(),
          centroids_d.data(),
          labels_d.data(),
          &inertia,
          &n_iter,
          &predict_inertia,
          &cluster_cost);

   ASSERT_TRUE(cuvs::devArrMatchHost(kExpectedCentroids,
                                     centroids_d.data(),
                                     kNClusters * kNFeatures,
                                     cuvs::CompareApprox<float>(1e-4f)));
   ASSERT_TRUE(cuvs::devArrMatchHost(
     kExpectedLabels, labels_d.data(), kNSamples, cuvs::Compare<int32_t>()));

   EXPECT_GT(n_iter, 0);
   EXPECT_NEAR(inertia, kExpectedInertia, 1e-4);
   EXPECT_NEAR(predict_inertia, kExpectedInertia, 1e-4);
   EXPECT_NEAR(cluster_cost, kExpectedInertia, 1e-4);
 }

 template <typename RunFn>
 void test_fit_host(RunFn run_fn)
 {
   raft::handle_t handle;
   auto stream = raft::resource::get_cuda_stream(handle);

   rmm::device_uvector<float> centroids_d(kNClusters * kNFeatures, stream);
   raft::copy(centroids_d.data(),
              reinterpret_cast<float const*>(kInitCentroids),
              kNClusters * kNFeatures,
              stream);

   double inertia         = -1.0;
   int n_iter             = -1;
   double unused_predict  = 0.0;
   double unused_cost     = 0.0;

   run_fn(kNSamples,
          kNFeatures,
          kNClusters,
          100,
          1e-6,
          Array,
          true,
          4,  // force at least 2 streamed batches
          reinterpret_cast<void*>(kDataset),
          centroids_d.data(),
          nullptr,
          &inertia,
          &n_iter,
          &unused_predict,
          &unused_cost);

   ASSERT_TRUE(cuvs::devArrMatchHost(kExpectedCentroids,
                                     centroids_d.data(),
                                     kNClusters * kNFeatures,
                                     cuvs::CompareApprox<float>(1e-4f)));

   EXPECT_GT(n_iter, 0);
   EXPECT_NEAR(inertia, kExpectedInertia, 1e-4);
 }

 }  // namespace

TEST(KMeansC, FitPredict) { test_fit_predict(run_kmeans); }
// TODO(cuVS 26.08): remove FitPredictV2 once `_v2` is promoted to the
// unsuffixed ABI -- it will be redundant with FitPredict at that point.
TEST(KMeansC, FitPredictV2) { test_fit_predict(run_kmeans_v2); }

TEST(KMeansC, FitHost) { test_fit_host(run_kmeans); }
// TODO(cuVS 26.08): remove FitHostV2 once `_v2` is promoted to the
// unsuffixed ABI.
TEST(KMeansC, FitHostV2) { test_fit_host(run_kmeans_v2); }

 TEST(KMeansC, ParamsCreateDestroy)
 {
   cuvsKMeansParams_t params = nullptr;
   ASSERT_EQ(cuvsKMeansParamsCreate(&params), CUVS_SUCCESS);
   ASSERT_NE(params, nullptr);
   EXPECT_GT(params->n_clusters, 0);
   EXPECT_GT(params->max_iter, 0);
   ASSERT_EQ(cuvsKMeansParamsDestroy(params), CUVS_SUCCESS);
 }

// TODO(cuVS 26.08): remove ParamsCreateDestroyV2 once cuvsKMeansParamsCreate_v2
// / cuvsKMeansParamsDestroy_v2 are promoted to the unsuffixed entry points and
// the `_v2` symbols are deleted from the public header.
TEST(KMeansC, ParamsCreateDestroyV2)
{
   cuvsKMeansParams_v2_t params = nullptr;
   ASSERT_EQ(cuvsKMeansParamsCreate_v2(&params), CUVS_SUCCESS);
   ASSERT_NE(params, nullptr);
   EXPECT_GT(params->n_clusters, 0);
   EXPECT_GT(params->max_iter, 0);
   ASSERT_EQ(cuvsKMeansParamsDestroy_v2(params), CUVS_SUCCESS);
 }
