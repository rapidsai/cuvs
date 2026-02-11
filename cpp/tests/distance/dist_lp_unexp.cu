/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs::distance {

template <typename DataType, typename OutputType = DataType>
class DistanceLpUnexp
  : public DistanceTest<cuvs::distance::DistanceType::LpUnexpanded, DataType, OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {

  {0.001f, 1024, 1024, 32, true, 1234ULL, 4.0f},
  {0.001f, 1024, 32, 1024, true, 1234ULL, 3.0f},
  {0.001f, 32, 1024, 1024, true, 1234ULL, 4.0f},
  {0.003f, 1024, 1024, 1024, true, 1234ULL, 3.0f},
  {0.001f, 1024, 1024, 32, false, 1234ULL, 4.0f},
  {0.001f, 1024, 32, 1024, false, 1234ULL, 3.0f},
  {0.001f, 32, 1024, 1024, false, 1234ULL, 4.0f},
  {0.003f, 1024, 1024, 1024, false, 1234ULL, 3.0f},
};
using DistanceLpUnexpF = DistanceLpUnexp<float>;
TEST_P(DistanceLpUnexpF,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceLpUnexpF,
  ::testing::ValuesIn(inputsf));  // NOLINT(modernize-use-trailing-return-type)

const std::vector<DistanceInputs<double>> inputsd = {

  {0.001, 1024, 1024, 32, true, 1234ULL, 4.0},
  {0.001, 1024, 32, 1024, true, 1234ULL, 3.0},
  {0.001, 32, 1024, 1024, true, 1234ULL, 4.0},
  {0.003, 1024, 1024, 1024, true, 1234ULL, 3.0},
  {0.001, 1024, 1024, 32, false, 1234ULL, 4.0},
  {0.001, 1024, 32, 1024, false, 1234ULL, 3.0},
  {0.001, 32, 1024, 1024, false, 1234ULL, 4.0},
  {0.003, 1024, 1024, 1024, false, 1234ULL, 3.0},
};
using DistanceLpUnexpD = DistanceLpUnexp<double>;
TEST_P(DistanceLpUnexpD,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceLpUnexpD,
  ::testing::ValuesIn(inputsd));  // NOLINT(modernize-use-trailing-return-type)

const std::vector<DistanceInputs<half, float>> inputsh = {

  {0.001f, 1024, 1024, 32, true, 1234ULL, 4.0f},
  {0.001f, 1024, 32, 1024, true, 1234ULL, 3.0f},
  {0.001f, 32, 1024, 1024, true, 1234ULL, 4.0f},
  {0.003f, 1024, 1024, 1024, true, 1234ULL, 3.0f},
  {0.001f, 1024, 1024, 32, false, 1234ULL, 4.0f},
  {0.001f, 1024, 32, 1024, false, 1234ULL, 3.0f},
  {0.001f, 32, 1024, 1024, false, 1234ULL, 4.0f},
  {0.003f, 1024, 1024, 1024, false, 1234ULL, 3.0f},
};
using DistanceLpUnexpH = DistanceLpUnexp<half, float>;
TEST_P(DistanceLpUnexpH,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceLpUnexpH,
  ::testing::ValuesIn(inputsh));  // NOLINT(modernize-use-trailing-return-type)

class BigMatrixLpUnexp : public BigMatrixDistanceTest<cuvs::distance::DistanceType::LpUnexpanded> {
};
TEST_F(BigMatrixLpUnexp, Result) {}  // NOLINT(modernize-use-trailing-return-type)
}  // namespace cuvs::distance
