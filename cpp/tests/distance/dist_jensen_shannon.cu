/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_base.cuh"

namespace cuvs::distance {

template <typename DataType, typename OutputType = DataType>
class DistanceJensenShannon
  : public DistanceTest<cuvs::distance::DistanceType::JensenShannon, DataType, OutputType> {};

const std::vector<DistanceInputs<float>> inputsf = {

  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceJensenShannonF = DistanceJensenShannon<float>;
TEST_P(DistanceJensenShannonF,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceJensenShannonF,
  ::testing::ValuesIn(inputsf));  // NOLINT(modernize-use-trailing-return-type)

const std::vector<DistanceInputs<double>> inputsd = {

  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001, 1024, 1024, 32, false, 1234ULL},
  {0.001, 1024, 32, 1024, false, 1234ULL},
  {0.001, 32, 1024, 1024, false, 1234ULL},
  {0.003, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceJensenShannonD = DistanceJensenShannon<double>;
TEST_P(DistanceJensenShannonD,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceJensenShannonD,
  ::testing::ValuesIn(inputsd));  // NOLINT(modernize-use-trailing-return-type)

const std::vector<DistanceInputs<half, float>> inputsh = {

  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
using DistanceJensenShannonH = DistanceJensenShannon<half, float>;
TEST_P(DistanceJensenShannonH,
       Result)  // NOLINT(modernize-use-trailing-return-type)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(cuvs::devArrMatch(
    dist_ref.data(), dist.data(), m, n, cuvs::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(
  DistanceTests,
  DistanceJensenShannonH,
  ::testing::ValuesIn(inputsh));  // NOLINT(modernize-use-trailing-return-type)

class BigMatrixJensenShannon
  : public BigMatrixDistanceTest<cuvs::distance::DistanceType::JensenShannon> {};
TEST_F(BigMatrixJensenShannon, Result) {}  // NOLINT(modernize-use-trailing-return-type)
}  // namespace cuvs::distance
