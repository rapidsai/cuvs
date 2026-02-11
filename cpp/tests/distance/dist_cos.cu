/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs::distance {

template <typename DataType, typename OutputType = DataType>
class DistanceExpCos
  : public DistanceTest<cuvs::distance::DistanceType::CosineExpanded, DataType, OutputType> {};

template <typename DataType, typename OutputType = DataType>
class DistanceExpCosXequalY
  : public DistanceTestSameBuffer<cuvs::distance::DistanceType::CosineExpanded,
                                  DataType,
                                  OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {

  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsXeqYf = {

  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsh = {

  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};

const std::vector<DistanceInputs<half, float>> inputsXeqYh = {

  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
};

using DistanceExpCosF = DistanceExpCos<float>;
TEST_P(DistanceExpCosF,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceExpCosF,
  ::testing::ValuesIn(inputsf));  // NOLINT(modernize-use-trailing-return-type)

using DistanceExpCosH = DistanceExpCos<half, float>;
TEST_P(DistanceExpCosH,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceExpCosH,
  ::testing::ValuesIn(inputsh));  // NOLINT(modernize-use-trailing-return-type)

using DistanceExpCosXequalYF = DistanceExpCosXequalY<float>;
TEST_P(DistanceExpCosXequalYF,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.m;
  int n = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                n,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  n = params.isRowMajor ? m : m / 2;
  m = params.isRowMajor ? m / 2 : m;

  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m,
                                n,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceExpCosXequalYF,
  ::testing::ValuesIn(inputsXeqYf));  // NOLINT(modernize-use-trailing-return-type)

using DistanceExpCosXequalYH = DistanceExpCosXequalY<half, float>;
TEST_P(DistanceExpCosXequalYH,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.m;
  int n = params.m;
  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                n,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
  n = params.isRowMajor ? m : m / 2;
  m = params.isRowMajor ? m / 2 : m;

  ASSERT_TRUE(cuvs::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m,
                                n,
                                cuvs::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceExpCosXequalYH,
  ::testing::ValuesIn(inputsXeqYh));  // NOLINT(modernize-use-trailing-return-type)

const std::vector<DistanceInputs<double>> inputsd = {

  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceExpCosD = DistanceExpCos<double>;
TEST_P(DistanceExpCosD,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceExpCosD,
  ::testing::ValuesIn(inputsd));  // NOLINT(modernize-use-trailing-return-type)

class BigMatrixCos : public BigMatrixDistanceTest<cuvs::distance::DistanceType::CosineExpanded> {};
TEST_F(BigMatrixCos, Result) {}  // NOLINT(modernize-use-trailing-return-type)

}  // namespace cuvs::distance
