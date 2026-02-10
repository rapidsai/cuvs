/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs {  // NOLINT(modernize-concat-nested-namespaces)
namespace distance {

template <typename DataType, typename OutputType = DataType>
class DistanceEucExpTest  // NOLINT(readability-identifier-naming)
  : public DistanceTest<cuvs::distance::DistanceType::L2Expanded, DataType, OutputType> {};

template <typename DataType, typename OutputType = DataType>
class DistanceEucExpTestXequalY  // NOLINT(readability-identifier-naming)
  : public DistanceTestSameBuffer<cuvs::distance::DistanceType::L2Expanded, DataType, OutputType> {
};

const std::vector<DistanceInputs<float>> inputsf = {
  // NOLINT(readability-identifier-naming)
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 2048, 4096, 128, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.003f, 1021, 1021, 1021, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
  {0.003f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsXeqYf = {
  // NOLINT(readability-identifier-naming)
  {0.01f, 2048, 4096, 128, true, 1234ULL},
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.03f, 1021, 1021, 1021, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
  {0.03f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsh = {
  // NOLINT(readability-identifier-naming)
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 2048, 4096, 128, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.003f, 1021, 1021, 1021, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
  {0.003f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsXeqYh = {
  // NOLINT(readability-identifier-naming)
  {0.01f, 2048, 4096, 128, true, 1234ULL},
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.03f, 1021, 1021, 1021, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
  {0.03f, 1021, 1021, 1021, false, 1234ULL},
};

using DistanceEucExpTestF = DistanceEucExpTest<float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceEucExpTestF,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceEucExpTestF,
  ::testing::ValuesIn(
    inputsf));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using DistanceEucExpTestH = DistanceEucExpTest<half, float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceEucExpTestH,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceEucExpTestH,
  ::testing::ValuesIn(
    inputsh));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

using DistanceEucExpTestXequalYF = DistanceEucExpTestXequalY<float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceEucExpTestXequalYF,
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
  DistanceTests,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  DistanceEucExpTestXequalYF,
  ::testing::ValuesIn(inputsXeqYf));

using DistanceEucExpTestXequalYH = DistanceEucExpTestXequalY<half, float>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceEucExpTestXequalYH,
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
  DistanceTests,  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
  DistanceEucExpTestXequalYH,
  ::testing::ValuesIn(inputsXeqYh));

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
using DistanceEucExpTestD = DistanceEucExpTest<double>;  // NOLINT(readability-identifier-naming)
TEST_P(DistanceEucExpTestD,
       Result)  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceEucExpTestD,
  ::testing::ValuesIn(
    inputsd));  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)

class BigMatrixEucExp : public BigMatrixDistanceTest<cuvs::distance::DistanceType::L2Expanded> {
};  // NOLINT(readability-identifier-naming)
TEST_F(BigMatrixEucExp, Result) {
}  // NOLINT(modernize-use-trailing-return-type,readability-identifier-naming)
}  // end namespace distance
}  // namespace cuvs
