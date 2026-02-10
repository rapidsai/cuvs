/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/stats/silhouette_score.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <random>

namespace cuvs {  // NOLINT(modernize-concat-nested-namespaces)
namespace stats {

// parameter structure definition
struct silhouetteScoreParam {  // NOLINT(readability-identifier-naming)
  int nRows;                   // NOLINT(readability-identifier-naming)
  int nCols;                   // NOLINT(readability-identifier-naming)
  int nLabels;                 // NOLINT(readability-identifier-naming)
  cuvs::distance::DistanceType metric;
  int chunk;
  double tolerance;
};

// test fixture class
template <typename LabelT, typename DataT>
class silhouetteScoreTest : public ::testing::TestWithParam<
                              silhouetteScoreParam> {  // NOLINT(readability-identifier-naming)
 protected:
  silhouetteScoreTest()
    : d_X(0, raft::resource::get_cuda_stream(handle)),
      sampleSilScore(0, raft::resource::get_cuda_stream(handle)),
      d_labels(0, raft::resource::get_cuda_stream(handle))
  {
  }

  void host_silhouette_score()
  {
    // generating random value test input
    std::vector<double> h_X(nElements, 0.0);  // NOLINT(readability-identifier-naming)
    std::vector<int> h_labels(nRows, 0);
    std::random_device rd;
    std::default_random_engine dre(nElements * nLabels);
    std::uniform_int_distribution<int> intGenerator(
      0, nLabels - 1);  // NOLINT(readability-identifier-naming)
    std::uniform_real_distribution<double> realGenerator(
      0, 100);  // NOLINT(readability-identifier-naming)

    std::generate(h_X.begin(), h_X.end(), [&]() -> auto { return realGenerator(dre); });
    std::generate(h_labels.begin(), h_labels.end(), [&]() -> auto { return intGenerator(dre); });

    // allocating and initializing memory to the GPU
    auto stream = raft::resource::get_cuda_stream(handle);
    d_X.resize(nElements, stream);
    d_labels.resize(nElements, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(d_X.data(), 0, d_X.size() * sizeof(DataT), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_labels.data(), 0, d_labels.size() * sizeof(LabelT), stream));
    sampleSilScore.resize(nElements, stream);

    raft::update_device(d_X.data(), &h_X[0], static_cast<int>(nElements), stream);
    raft::update_device(d_labels.data(), &h_labels[0], static_cast<int>(nElements), stream);

    // finding the distance matrix

    rmm::device_uvector<double> d_distanceMatrix(nRows * nRows,
                                                 stream);  // NOLINT(readability-identifier-naming)
    double* h_distanceMatrix =
      (double*)malloc(nRows * nRows * sizeof(double*));  // NOLINT(readability-identifier-naming)

    auto d_X_view = raft::make_device_matrix_view<const DataT, int64_t>(
      d_X.data(), nRows, nCols);  // NOLINT(readability-identifier-naming)
    cuvs::distance::pairwise_distance(
      handle,
      d_X_view,
      d_X_view,
      raft::make_device_matrix_view<DataT, int64_t>(d_distanceMatrix.data(), nRows, nRows),
      params.metric);

    raft::resource::sync_stream(handle, stream);

    raft::update_host(h_distanceMatrix, d_distanceMatrix.data(), nRows * nRows, stream);

    // finding the bincount array

    double* binCountArray =
      (double*)malloc(nLabels * sizeof(double*));  // NOLINT(readability-identifier-naming)
    memset(binCountArray, 0, nLabels * sizeof(double));

    for (int i = 0; i < nRows; ++i) {
      binCountArray[h_labels[i]] += 1;
    }

    // finding the average intra cluster distance for every element

    double* a = (double*)malloc(nRows * sizeof(double*));

    for (int i = 0; i < nRows; ++i) {
      int myLabel               = h_labels[i];  // NOLINT(readability-identifier-naming)
      double sumOfIntraClusterD = 0;            // NOLINT(readability-identifier-naming)

      for (int j = 0; j < nRows; ++j) {
        if (h_labels[j] == myLabel) { sumOfIntraClusterD += h_distanceMatrix[i * nRows + j]; }
      }

      if (binCountArray[myLabel] <= 1)  // NOLINT(google-readability-braces-around-statements)
        a[i] = -1;
      else  // NOLINT(google-readability-braces-around-statements)
        a[i] = sumOfIntraClusterD / (binCountArray[myLabel] - 1);
    }

    // finding the average inter cluster distance for every element

    double* b = (double*)malloc(nRows * sizeof(double*));

    for (int i = 0; i < nRows; ++i) {
      int myLabel          = h_labels[i];  // NOLINT(readability-identifier-naming)
      double minAvgInterCD = ULLONG_MAX;   // NOLINT(readability-identifier-naming)

      for (int j = 0; j < nLabels; ++j) {
        int curClLabel = j;  // NOLINT(readability-identifier-naming)
        if (curClLabel == myLabel) continue;
        double avgInterCD = 0;  // NOLINT(readability-identifier-naming)

        for (int k = 0; k < nRows; ++k) {
          if (h_labels[k] == curClLabel) { avgInterCD += h_distanceMatrix[i * nRows + k]; }
        }

        if (binCountArray[curClLabel])  // NOLINT(google-readability-braces-around-statements)
          avgInterCD /= binCountArray[curClLabel];
        else  // NOLINT(google-readability-braces-around-statements)
          avgInterCD = ULLONG_MAX;
        minAvgInterCD = min(minAvgInterCD, avgInterCD);
      }

      b[i] = minAvgInterCD;
    }

    // finding the silhouette score for every element

    double* truthSampleSilScore =
      (double*)malloc(nRows * sizeof(double*));  // NOLINT(readability-identifier-naming)
    for (int i = 0; i < nRows; ++i) {
      if (a[i] == -1)  // NOLINT(google-readability-braces-around-statements)
        truthSampleSilScore[i] = 0;
      else if (a[i] == 0 && b[i] == 0)  // NOLINT(google-readability-braces-around-statements)
        truthSampleSilScore[i] = 0;
      else  // NOLINT(google-readability-braces-around-statements)
        truthSampleSilScore[i] = (b[i] - a[i]) / max(a[i], b[i]);
      truthSilhouetteScore += truthSampleSilScore[i];
    }

    truthSilhouetteScore /= nRows;
  }

  // the constructor
  void SetUp() override  // NOLINT(readability-identifier-naming)
  {
    // getting the parameters
    params = ::testing::TestWithParam<silhouetteScoreParam>::GetParam();

    nRows     = params.nRows;
    nCols     = params.nCols;
    nLabels   = params.nLabels;
    chunk     = params.chunk;
    nElements = nRows * nCols;

    host_silhouette_score();

    // calling the silhouette_score CUDA implementation
    computedSilhouetteScore = cuvs::stats::silhouette_score(
      handle,
      raft::make_device_matrix_view<const DataT>(d_X.data(), nRows, nCols),
      raft::make_device_vector_view<const LabelT>(d_labels.data(), nRows),
      std::make_optional(raft::make_device_vector_view(sampleSilScore.data(), nRows)),
      nLabels,
      params.metric);

    batchedSilhouetteScore = cuvs::stats::silhouette_score_batched(
      handle,
      raft::make_device_matrix_view<const DataT>(d_X.data(), nRows, nCols),
      raft::make_device_vector_view<const LabelT>(d_labels.data(), nRows),
      std::make_optional(raft::make_device_vector_view(sampleSilScore.data(), nRows)),
      nLabels,
      chunk,
      params.metric);
  }

  // declaring the data values
  raft::resources handle;                     // NOLINT(readability-identifier-naming)
  silhouetteScoreParam params;                // NOLINT(readability-identifier-naming)
  int nLabels;                                // NOLINT(readability-identifier-naming)
  rmm::device_uvector<DataT> d_X;             // NOLINT(readability-identifier-naming)
  rmm::device_uvector<DataT> sampleSilScore;  // NOLINT(readability-identifier-naming)
  rmm::device_uvector<LabelT> d_labels;       // NOLINT(readability-identifier-naming)
  int nRows;                                  // NOLINT(readability-identifier-naming)
  int nCols;                                  // NOLINT(readability-identifier-naming)
  int nElements;                              // NOLINT(readability-identifier-naming)
  double truthSilhouetteScore    = 0;         // NOLINT(readability-identifier-naming)
  double computedSilhouetteScore = 0;         // NOLINT(readability-identifier-naming)
  double batchedSilhouetteScore  = 0;         // NOLINT(readability-identifier-naming)
  int chunk;                                  // NOLINT(readability-identifier-naming)
};

// setting test parameter values
const std::vector<silhouetteScoreParam> inputs = {  // NOLINT(readability-identifier-naming)
  {4, 2, 3, cuvs::distance::DistanceType::L2Expanded, 4, 0.00001},
  {4, 2, 2, cuvs::distance::DistanceType::L2SqrtUnexpanded, 2, 0.00001},
  {8, 8, 3, cuvs::distance::DistanceType::L2Unexpanded, 4, 0.00001},
  {11, 2, 5, cuvs::distance::DistanceType::L2Expanded, 3, 0.00001},
  {40, 2, 8, cuvs::distance::DistanceType::L2Expanded, 10, 0.00001},
  {12, 7, 3, cuvs::distance::DistanceType::CosineExpanded, 8, 0.00001},
  {7, 5, 5, cuvs::distance::DistanceType::L1, 2, 0.00001}};

// writing the test suite
using silhouetteScoreTestClass =
  silhouetteScoreTest<int, double>;  // NOLINT(readability-identifier-naming)
TEST_P(silhouetteScoreTestClass,
       Result)  // NOLINT(readability-identifier-naming)
{
  ASSERT_NEAR(computedSilhouetteScore, truthSilhouetteScore, params.tolerance);
  ASSERT_NEAR(batchedSilhouetteScore, truthSilhouetteScore, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(silhouetteScore,
                        silhouetteScoreTestClass,
                        ::testing::ValuesIn(inputs));  // NOLINT(readability-identifier-naming)

}  // end namespace stats
}  // end namespace cuvs
