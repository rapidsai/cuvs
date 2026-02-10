/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/cluster/kmeans.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/stats/adjusted_rand_index.cuh>

#include <raft/comms/std_comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <gtest/gtest.h>
#include <nccl.h>
#include <stdio.h>  // NOLINT(modernize-deprecated-headers)
#include <test_utils.h>

#include <vector>

#define NCCLCHECK(cmd)                                                                        \
  do {                                                                                        \
    ncclResult_t res = cmd;                                                                   \
    if (res != ncclSuccess) {                                                                 \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
      exit(EXIT_FAILURE);                                                                     \
    }                                                                                         \
  } while (0)

namespace cuvs {

template <typename T>
struct KmeansInputs {  // NOLINT(readability-identifier-naming)
  int n_row;
  int n_col;
  int n_clusters;
  T tol;
  bool weighted;
};

template <typename T>
class KmeansTest  // NOLINT(readability-identifier-naming)
  : public ::testing::TestWithParam<KmeansInputs<T>> {
 protected:
  KmeansTest()  // NOLINT(modernize-use-equals-default)
    : stream(handle.get_stream()),
      d_labels(0, stream),
      d_labels_ref(0, stream),
      d_centroids(0, stream),
      d_sample_weight(0, stream)
  {
  }

  void basicTest()  // NOLINT(readability-identifier-naming)
  {
    testparams = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
    ncclComm_t nccl_comm;
    NCCLCHECK(ncclCommInitAll(&nccl_comm, 1, {0}));  // NOLINT(modernize-use-nullptr)
    raft::comms::build_comms_nccl_only(&handle, nccl_comm, 1, 0);

    int n_samples              = testparams.n_row;
    int n_features             = testparams.n_col;
    params.n_clusters          = testparams.n_clusters;
    params.tol                 = testparams.tol;
    params.n_init              = 5;
    params.rng_state.seed      = 1;
    params.oversampling_factor = 1;

    auto stream = handle.get_stream();
    rmm::device_uvector<T> x(n_samples * n_features, stream);
    rmm::device_uvector<int> labels(n_samples, stream);

    raft::random::make_blobs<T, int>(handle,
                                     x.data(),
                                     labels.data(),
                                     n_samples,
                                     n_features,
                                     params.n_clusters,
                                     true,
                                     nullptr,
                                     nullptr,
                                     1.0,
                                     false,
                                     -10.0f,
                                     10.0f,
                                     1234ULL);

    d_labels.resize(n_samples, stream);
    d_labels_ref.resize(n_samples, stream);
    d_centroids.resize(params.n_clusters * n_features, stream);

    std::optional<raft::device_vector_view<const T, int>> d_sw = std::nullopt;
    if (testparams.weighted) {
      d_sample_weight.resize(n_samples, stream);
      thrust::fill(thrust::cuda::par.on(stream),
                   d_sample_weight.data(),
                   d_sample_weight.data() + n_samples,
                   1);
      d_sw = raft::make_device_vector_view<const T, int>(d_sample_weight.data(), n_samples);
    }
    raft::copy(d_labels_ref.data(), labels.data(), n_samples, stream);

    handle.sync_stream(stream);

    T inertia  = 0;
    int n_iter = 0;

    auto x_view = raft::make_device_matrix_view<const T, int>(x.data(), n_samples, n_features);
    auto centroids_view =
      raft::make_device_matrix_view<T, int>(d_centroids.data(), params.n_clusters, n_features);

    cuvs::cluster::kmeans::fit(handle,
                               params,
                               x_view,
                               d_sw,
                               centroids_view,
                               raft::make_host_scalar_view<T>(&inertia),
                               raft::make_host_scalar_view<int>(&n_iter));

    cuvs::cluster::kmeans::predict(
      handle,
      params,
      x_view,
      d_sw,
      d_centroids.data(),
      raft::make_device_vector_view<int, int>(d_labels.data(), n_samples),
      true,
      raft::make_host_scalar_view<T>(&inertia));
    score = raft::stats::adjusted_rand_index(
      d_labels_ref.data(), d_labels.data(), n_samples, raft::resource::get_cuda_stream(handle));
    handle.sync_stream(stream);

    if (score < 0.99) {
      std::cout << "Expected: " << raft::arr2Str(d_labels_ref.data(), 25, "d_labels_ref", stream)
                << std::endl;
      std::cout << "Actual: " << raft::arr2Str(d_labels.data(), 25, "d_labels", stream)
                << std::endl;
      std::cout << "score = " << score << std::endl;
    }
    ncclCommDestroy(nccl_comm);
  }

  void SetUp() override { basicTest(); }  // NOLINT(readability-identifier-naming)

 protected:
  raft::handle_t handle;                   // NOLINT(readability-identifier-naming)
  cudaStream_t stream;                     // NOLINT(readability-identifier-naming)
  KmeansInputs<T> testparams;              // NOLINT(readability-identifier-naming)
  rmm::device_uvector<int> d_labels;       // NOLINT(readability-identifier-naming)
  rmm::device_uvector<int> d_labels_ref;   // NOLINT(readability-identifier-naming)
  rmm::device_uvector<T> d_centroids;      // NOLINT(readability-identifier-naming)
  rmm::device_uvector<T> d_sample_weight;  // NOLINT(readability-identifier-naming)
  double score;                            // NOLINT(readability-identifier-naming)
  cuvs::cluster::kmeans::params params;    // NOLINT(readability-identifier-naming)
};

const std::vector<KmeansInputs<float>> kInputsf2 = {  // NOLINT(readability-identifier-naming)
  {1000, 32, 5, 0.0001, true},
  {1000, 32, 5, 0.0001, false},
  {1000, 100, 20, 0.0001, true},
  {1000, 100, 20, 0.0001, false},
  {10000, 32, 10, 0.0001, true},
  {10000, 32, 10, 0.0001, false},
  {10000, 100, 50, 0.0001, true},
  {10000, 100, 50, 0.0001, false}};

const std::vector<KmeansInputs<double>> kInputsd2 = {  // NOLINT(readability-identifier-naming)
  {1000, 32, 5, 0.0001, true},
  {1000, 32, 5, 0.0001, false},
  {1000, 100, 20, 0.0001, true},
  {1000, 100, 20, 0.0001, false},
  {10000, 32, 10, 0.0001, true},
  {10000, 32, 10, 0.0001, false},
  {10000, 100, 50, 0.0001, true},
  {10000, 100, 50, 0.0001, false}};

using KmeansTestF = KmeansTest<float>;  // NOLINT(readability-identifier-naming)
TEST_P(KmeansTestF, Result)
{
  ASSERT_TRUE(score >= 0.99);
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using KmeansTestD = KmeansTest<double>;  // NOLINT(readability-identifier-naming)
TEST_P(KmeansTestD, Result)
{
  ASSERT_TRUE(score >= 0.99);
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  KmeansTests,
  KmeansTestF,
  ::testing::ValuesIn(
    kInputsf2));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

INSTANTIATE_TEST_CASE_P(
  KmeansTests,
  KmeansTestD,
  ::testing::ValuesIn(
    kInputsd2));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

}  // end namespace cuvs
