/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs {  // NOLINT(modernize-concat-nested-namespaces)
namespace distance {

template <typename DataType, typename OutputType = DataType>
class DistanceCorrelation  // NOLINT(readability-identifier-naming)
  : public DistanceTest<cuvs::distance::DistanceType::CorrelationExpanded, DataType, OutputType> {};

template <typename DataType, typename OutputType = DataType>
class DistanceCorrelationXequalY  // NOLINT(readability-identifier-naming)
  : public DistanceTestSameBuffer<cuvs::distance::DistanceType::CorrelationExpanded,
                                  DataType,
                                  OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  // NOLINT(readability-identifier-naming)
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceCorrelationF = DistanceCorrelation<float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceCorrelationF,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceCorrelationF,
  ::testing::ValuesIn(
    inputsf));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

const std::vector<DistanceInputs<half, float>> inputsh = {
  // NOLINT(readability-identifier-naming)
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceCorrelationH = DistanceCorrelation<half, float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceCorrelationH,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceCorrelationH,
  ::testing::ValuesIn(
    inputsh));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using DistanceCorrelationXequalYF = DistanceCorrelationXequalY<float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceCorrelationXequalYF,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m / 2,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceCorrelationXequalYF,
  ::testing::ValuesIn(
    inputsf));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

const std::vector<DistanceInputs<double>> inputsd = {
  // NOLINT(readability-identifier-naming)
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001, 1024, 1024, 32, false, 1234ULL},
  {0.001, 1024, 32, 1024, false, 1234ULL},
  {0.001, 32, 1024, 1024, false, 1234ULL},
  {0.003, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceCorrelationD = DistanceCorrelation<double>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceCorrelationD,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceCorrelationD,
  ::testing::ValuesIn(
    inputsd));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using DistanceCorrelationXequalYH = DistanceCorrelationXequalY<half, float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceCorrelationXequalYH,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m / 2,
                                m,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceCorrelationXequalYH,
  ::testing::ValuesIn(
    inputsh));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

class BigMatrixCorrelation  // NOLINT(readability-identifier-naming)
  : public BigMatrixDistanceTest<cuvs::distance::DistanceType::CorrelationExpanded> {};
TEST_F(BigMatrixCorrelation, Result) {
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
}  // end namespace distance
}  // namespace cuvs
